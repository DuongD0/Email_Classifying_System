import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import * as kv from "./kv_store.tsx";

const app = new Hono();

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Health check endpoint
app.get("/make-server-4ca42d89/health", (c) => {
  return c.json({ status: "ok" });
});

// Gmail integration endpoints
app.get("/make-server-4ca42d89/gmail/status", async (c) => {
  try {
    const accessToken = await kv.get("gmail_access_token");
    return c.json({ connected: !!accessToken });
  } catch (error) {
    console.log("Gmail status check error:", error);
    return c.json({ connected: false });
  }
});

app.post("/make-server-4ca42d89/gmail/auth", async (c) => {
  try {
    const clientId = Deno.env.get("GOOGLE_CLIENT_ID");
    const redirectUri = Deno.env.get("GOOGLE_REDIRECT_URI") || "http://localhost:3000/gmail-callback";
    
    if (!clientId) {
      return c.json({ error: "Google Client ID not configured" }, 400);
    }
    
    const scopes = [
      "https://www.googleapis.com/auth/gmail.readonly",
      "https://www.googleapis.com/auth/userinfo.email"
    ].join(" ");
    
    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${clientId}&` +
      `redirect_uri=${encodeURIComponent(redirectUri)}&` +
      `scope=${encodeURIComponent(scopes)}&` +
      `response_type=code&` +
      `access_type=offline&` +
      `prompt=consent`;
    
    return c.json({ authUrl });
  } catch (error) {
    console.log("Gmail auth error:", error);
    return c.json({ error: "Failed to generate auth URL" }, 500);
  }
});

app.post("/make-server-4ca42d89/gmail/callback", async (c) => {
  try {
    const { code } = await c.req.json();
    const clientId = Deno.env.get("GOOGLE_CLIENT_ID");
    const clientSecret = Deno.env.get("GOOGLE_CLIENT_SECRET");
    const redirectUri = Deno.env.get("GOOGLE_REDIRECT_URI") || "http://localhost:3000/gmail-callback";
    
    if (!clientId || !clientSecret) {
      return c.json({ error: "Google credentials not configured" }, 400);
    }
    
    const tokenResponse = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        code,
        client_id: clientId,
        client_secret: clientSecret,
        redirect_uri: redirectUri,
        grant_type: "authorization_code",
      }),
    });
    
    if (!tokenResponse.ok) {
      throw new Error("Failed to exchange code for tokens");
    }
    
    const tokens = await tokenResponse.json();
    
    // Store tokens
    await kv.set("gmail_access_token", tokens.access_token);
    if (tokens.refresh_token) {
      await kv.set("gmail_refresh_token", tokens.refresh_token);
    }
    
    return c.json({ success: true });
  } catch (error) {
    console.log("Gmail callback error:", error);
    return c.json({ error: "Failed to process authentication" }, 500);
  }
});

app.get("/make-server-4ca42d89/gmail/emails", async (c) => {
  try {
    const accessToken = await kv.get("gmail_access_token");
    if (!accessToken) {
      return c.json({ error: "Not authenticated with Gmail" }, 401);
    }
    
    const response = await fetch(
      "https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=20&q=in:inbox",
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    
    if (!response.ok) {
      throw new Error("Failed to fetch emails from Gmail API");
    }
    
    const data = await response.json();
    const emails = [];
    
    // Fetch details for each email
    for (const message of data.messages || []) {
      try {
        const emailResponse = await fetch(
          `https://gmail.googleapis.com/gmail/v1/users/me/messages/${message.id}`,
          {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          }
        );
        
        if (emailResponse.ok) {
          const emailData = await emailResponse.json();
          const headers = emailData.payload.headers;
          
          const getHeader = (name: string) => 
            headers.find((h: any) => h.name.toLowerCase() === name.toLowerCase())?.value || "";
          
          emails.push({
            id: emailData.id,
            threadId: emailData.threadId,
            subject: getHeader("Subject"),
            sender: getHeader("From"),
            snippet: emailData.snippet,
            date: new Date(parseInt(emailData.internalDate)).toISOString(),
            isRead: !emailData.labelIds?.includes("UNREAD"),
          });
        }
      } catch (emailError) {
        console.log("Error fetching individual email:", emailError);
      }
    }
    
    return c.json({ emails });
  } catch (error) {
    console.log("Gmail emails fetch error:", error);
    return c.json({ error: "Failed to fetch emails" }, 500);
  }
});

app.post("/make-server-4ca42d89/gmail/analyze/:emailId", async (c) => {
  try {
    const emailId = c.req.param("emailId");
    const accessToken = await kv.get("gmail_access_token");
    
    if (!accessToken) {
      return c.json({ error: "Not authenticated with Gmail" }, 401);
    }
    
    // Fetch full email content
    const response = await fetch(
      `https://gmail.googleapis.com/gmail/v1/users/me/messages/${emailId}?format=full`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    
    if (!response.ok) {
      throw new Error("Failed to fetch email details");
    }
    
    const emailData = await response.json();
    const headers = emailData.payload.headers;
    const getHeader = (name: string) => 
      headers.find((h: any) => h.name.toLowerCase() === name.toLowerCase())?.value || "";
    
    // Extract email content
    let emailContent = "";
    if (emailData.payload.body?.data) {
      emailContent = atob(emailData.payload.body.data.replace(/-/g, '+').replace(/_/g, '/'));
    } else if (emailData.payload.parts) {
      for (const part of emailData.payload.parts) {
        if (part.mimeType === "text/plain" && part.body?.data) {
          emailContent += atob(part.body.data.replace(/-/g, '+').replace(/_/g, '/'));
        }
      }
    }
    
    // Simple risk analysis based on keywords and patterns
    const suspiciousKeywords = [
      'urgent', 'limited time', 'act now', 'click here', 'verify account',
      'suspended', 'winner', 'congratulations', 'free money', 'lottery',
      'inheritance', 'prince', 'million dollars', 'tax refund'
    ];
    
    const emailText = (getHeader("Subject") + " " + emailContent).toLowerCase();
    let riskScore = 0;
    let matchedKeywords = 0;
    
    suspiciousKeywords.forEach(keyword => {
      if (emailText.includes(keyword)) {
        matchedKeywords++;
        riskScore += 15;
      }
    });
    
    // Check sender domain
    const sender = getHeader("From");
    const domain = sender.split('@')[1]?.toLowerCase() || "";
    
    // Simple domain checks
    const suspiciousDomains = ['gmail.com', 'yahoo.com', 'hotmail.com'];
    if (suspiciousDomains.some(d => domain.includes(d)) && matchedKeywords > 0) {
      riskScore += 10;
    }
    
    // Check for authentication failures
    const authResults = getHeader("Authentication-Results");
    if (authResults.includes("spf=fail") || authResults.includes("dkim=fail")) {
      riskScore += 20;
    }
    
    riskScore = Math.min(riskScore, 100);
    
    let riskLevel: 'low' | 'medium' | 'high';
    if (riskScore >= 70) riskLevel = 'high';
    else if (riskScore >= 40) riskLevel = 'medium';
    else riskLevel = 'low';
    
    // Store analysis result
    await kv.set(`email_analysis_${emailId}`, {
      riskScore,
      riskLevel,
      matchedKeywords,
      analyzedAt: new Date().toISOString(),
      sender,
      subject: getHeader("Subject"),
      content: emailContent.substring(0, 1000), // Store first 1000 chars
    });
    
    return c.json({
      riskScore,
      riskLevel,
      matchedKeywords,
      analysis: {
        sender,
        subject: getHeader("Subject"),
        suspiciousElements: matchedKeywords,
        authenticationIssues: authResults.includes("fail"),
      }
    });
  } catch (error) {
    console.log("Gmail analysis error:", error);
    return c.json({ error: "Failed to analyze email" }, 500);
  }
});

app.get("/make-server-4ca42d89/gmail/email/:emailId", async (c) => {
  try {
    const emailId = c.req.param("emailId");
    
    // Get stored analysis
    const analysis = await kv.get(`email_analysis_${emailId}`);
    if (!analysis) {
      return c.json({ error: "Email analysis not found" }, 404);
    }
    
    const accessToken = await kv.get("gmail_access_token");
    if (!accessToken) {
      return c.json({ error: "Not authenticated with Gmail" }, 401);
    }
    
    // Fetch full email details for detailed analysis
    const response = await fetch(
      `https://gmail.googleapis.com/gmail/v1/users/me/messages/${emailId}?format=full`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );
    
    if (!response.ok) {
      throw new Error("Failed to fetch email details");
    }
    
    const emailData = await response.json();
    const headers = emailData.payload.headers;
    
    return c.json({
      ...analysis,
      headers: headers.reduce((acc: any, header: any) => {
        acc[header.name] = header.value;
        return acc;
      }, {}),
      fullContent: emailData
    });
  } catch (error) {
    console.log("Gmail email detail error:", error);
    return c.json({ error: "Failed to fetch email details" }, 500);
  }
});

Deno.serve(app.fetch);