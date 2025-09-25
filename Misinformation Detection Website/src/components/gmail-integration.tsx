import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Mail, RefreshCw, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { projectId, publicAnonKey } from '../utils/supabase/info';

interface Email {
  id: string;
  threadId: string;
  subject: string;
  sender: string;
  snippet: string;
  date: string;
  isRead: boolean;
  riskScore?: number;
  riskLevel?: 'low' | 'medium' | 'high';
}

interface GmailIntegrationProps {
  onEmailSelect: (email: Email) => void;
}

export function GmailIntegration({ onEmailSelect }: GmailIntegrationProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [emails, setEmails] = useState<Email[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkGmailConnection();
  }, []);

  const checkGmailConnection = async () => {
    try {
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/status`, {
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setIsConnected(data.connected);
        if (data.connected) {
          fetchEmails();
        }
      }
    } catch (err) {
      console.error('Failed to check Gmail connection:', err);
    }
  };

  const connectGmail = async () => {
    try {
      setLoading(true);
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/auth`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        window.location.href = data.authUrl;
      } else {
        throw new Error('Failed to initiate Gmail authentication');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to Gmail');
    } finally {
      setLoading(false);
    }
  };

  const fetchEmails = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/emails`, {
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setEmails(data.emails);
      } else {
        throw new Error('Failed to fetch emails');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch emails');
    } finally {
      setLoading(false);
    }
  };

  const analyzeEmail = async (emailId: string) => {
    try {
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/analyze/${emailId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
        },
      });
      
      if (response.ok) {
        const analysis = await response.json();
        const updatedEmails = emails.map(email => 
          email.id === emailId 
            ? { ...email, riskScore: analysis.riskScore, riskLevel: analysis.riskLevel }
            : email
        );
        setEmails(updatedEmails);
        
        const emailToAnalyze = updatedEmails.find(email => email.id === emailId);
        if (emailToAnalyze) {
          onEmailSelect(emailToAnalyze);
        }
      }
    } catch (err) {
      console.error('Failed to analyze email:', err);
    }
  };

  const getRiskBadgeColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (!isConnected) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Mail className="h-5 w-5" />
            <span>Gmail Integration</span>
          </CardTitle>
          <CardDescription>
            Connect your Gmail account to analyze your emails for potential threats
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              To analyze your emails, you need to grant access to your Gmail account. 
              We only read email metadata and content for security analysis.
            </AlertDescription>
          </Alert>
          
          <div className="space-y-3">
            <h4 className="font-medium">What we analyze:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Email content for suspicious keywords and patterns</li>
              <li>• Sender authentication (SPF, DKIM, DMARC)</li>
              <li>• Domain reputation and age</li>
              <li>• Links and attachments for potential threats</li>
            </ul>
          </div>

          <Button 
            onClick={connectGmail} 
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <Mail className="h-4 w-4 mr-2" />
                Connect Gmail Account
              </>
            )}
          </Button>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <span>Gmail Connected</span>
              </CardTitle>
              <CardDescription>
                Select an email below to perform detailed security analysis
              </CardDescription>
            </div>
            <Button onClick={fetchEmails} disabled={loading} variant="outline">
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        {emails.length === 0 && !loading ? (
          <Card>
            <CardContent className="flex items-center justify-center py-8">
              <p className="text-muted-foreground">No emails found</p>
            </CardContent>
          </Card>
        ) : (
          emails.map((email) => (
            <Card key={email.id} className="cursor-pointer hover:bg-accent/50 transition-colors">
              <CardContent className="p-4" onClick={() => analyzeEmail(email.id)}>
                <div className="flex items-start justify-between space-x-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <div className={`w-2 h-2 rounded-full ${email.isRead ? 'bg-gray-400' : 'bg-blue-500'}`} />
                      <p className="font-medium truncate">{email.subject || '(No Subject)'}</p>
                      {email.riskLevel && (
                        <Badge className={getRiskBadgeColor(email.riskLevel)}>
                          {email.riskLevel} risk
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-1">{email.sender}</p>
                    <p className="text-sm text-muted-foreground truncate">{email.snippet}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{new Date(email.date).toLocaleString()}</span>
                      </div>
                      {email.riskScore && (
                        <span>Risk Score: {email.riskScore}%</span>
                      )}
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    Analyze
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="h-6 w-6 animate-spin mr-2" />
          <span>Loading emails...</span>
        </div>
      )}
    </div>
  );
}