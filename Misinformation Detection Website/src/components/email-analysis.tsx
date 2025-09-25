import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { AlertTriangle, Mail, Clock, User, Globe, Shield, Eye, Flag } from 'lucide-react';
import { useState } from 'react';
import { GmailIntegration } from './gmail-integration';

const suspiciousWords = [
  { word: 'URGENT', count: 3, riskLevel: 'high', positions: [12, 89, 156] },
  { word: 'limited time', count: 2, riskLevel: 'high', positions: [45, 234] },
  { word: 'click here', count: 4, riskLevel: 'medium', positions: [67, 123, 178, 298] },
  { word: 'verify', count: 2, riskLevel: 'medium', positions: [189, 267] },
  { word: 'suspended', count: 1, riskLevel: 'high', positions: [201] },
  { word: 'act now', count: 1, riskLevel: 'high', positions: [312] }
];

const classificationBreakdown = [
  { name: 'Normal Content', value: 25, color: '#22c55e' },
  { name: 'Spam Indicators', value: 45, color: '#f59e0b' },
  { name: 'Fraud Indicators', value: 30, color: '#ef4444' }
];

const sampleEmail = {
  subject: 'URGENT: Your Account Will Be Suspended - Verify Now!',
  sender: 'security@fake-bank-alert.com',
  receivedDate: '2024-09-23T10:30:00Z',
  riskScore: 87,
  content: `Dear Valued Customer,

URGENT NOTICE: We have detected suspicious activity on your account. Your account will be suspended within 24 hours if you do not verify your information immediately.

To prevent account suspension, click here to verify your account details:
[Verify Account Now - Limited Time]

This is a time-sensitive matter. Please act now to avoid any disruption to your banking services.

If you do not verify within the next few hours, your account will be permanently suspended and you may lose access to your funds.

Thank you for your immediate attention.

Security Department
Fake Bank Alert Team`,
  headers: {
    'Return-Path': 'bounce@fake-bank-alert.com',
    'X-Originating-IP': '192.168.1.100',
    'Authentication-Results': 'spf=fail smtp.mailfrom=fake-bank-alert.com',
    'DKIM-Signature': 'none'
  },
  technicalAnalysis: {
    spfCheck: 'FAIL',
    dkimCheck: 'FAIL',
    dmarcCheck: 'FAIL',
    domainAge: '2 days',
    sslCertificate: 'Invalid',
    reputation: 'Blacklisted'
  }
};

const wordAnalysisData = suspiciousWords.map(word => ({
  word: word.word,
  count: word.count,
  risk: word.riskLevel === 'high' ? 3 : word.riskLevel === 'medium' ? 2 : 1
}));

interface SelectedEmail {
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

interface EmailAnalysisProps {
  isDemoMode?: boolean;
}

export function EmailAnalysis({ isDemoMode = false }: EmailAnalysisProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedEmail, setSelectedEmail] = useState<SelectedEmail | null>(null);
  const [emailDetails, setEmailDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleEmailSelect = async (email: SelectedEmail) => {
    setSelectedEmail(email);
    setLoading(true);
    
    try {
      const { projectId, publicAnonKey } = await import('../utils/supabase/info');
      const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/email/${email.id}`, {
        headers: {
          'Authorization': `Bearer ${publicAnonKey}`,
        },
      });
      
      if (response.ok) {
        const details = await response.json();
        setEmailDetails(details);
      }
    } catch (error) {
      console.error('Failed to fetch email details:', error);
    } finally {
      setLoading(false);
    }
  };

  const highlightSuspiciousWords = (content: string) => {
    let highlightedContent = content;
    suspiciousWords.forEach(({ word, riskLevel }) => {
      const className = riskLevel === 'high' 
        ? 'bg-red-200 text-red-900 px-1 rounded font-medium' 
        : 'bg-yellow-200 text-yellow-900 px-1 rounded font-medium';
      
      const regex = new RegExp(`(${word})`, 'gi');
      highlightedContent = highlightedContent.replace(
        regex, 
        `<span class="${className}">$1</span>`
      );
    });
    return highlightedContent;
  };

  // If no email is selected, show appropriate content based on mode
  if (!selectedEmail) {
    if (isDemoMode) {
      // For demo mode, show a sample email analysis automatically
      const demoEmail = {
        id: 'demo-123',
        threadId: 'demo-thread-123',
        subject: sampleEmail.subject,
        sender: sampleEmail.sender,
        snippet: 'URGENT NOTICE: We have detected suspicious activity on your account...',
        date: sampleEmail.receivedDate,
        isRead: false,
        riskScore: sampleEmail.riskScore,
        riskLevel: 'high' as const
      };
      
      return (
        <div className="p-6 space-y-6 max-w-7xl mx-auto">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Email Analysis</h1>
            <p className="text-muted-foreground">
              Demo analysis of a fraudulent email sample
            </p>
          </div>
          
          <Card className="border-blue-200 bg-blue-50">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="font-medium text-blue-800">Demo Mode</p>
                  <p className="text-sm text-blue-700">
                    This is a demonstration using sample fraud email data. In the real application, 
                    you would connect your Gmail account to analyze your actual emails.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Show demo analysis */}
          {(() => {
            // Set the demo email as selected to show the analysis
            const currentEmail = demoEmail;
            const currentDetails = { content: sampleEmail.content };

            return (
              <>
                {/* Email Overview */}
                <Card className="border-l-4 border-l-red-500">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <CardTitle className="text-xl text-red-700">
                          Fraud Email Detected
                        </CardTitle>
                        <CardDescription>
                          This email has been classified as fraudulent
                        </CardDescription>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge 
                          variant="destructive"
                          className="text-lg px-3 py-1 bg-red-100 text-red-800"
                        >
                          Confidence: {currentEmail.riskScore}%
                        </Badge>
                        <Button variant="outline" size="sm">
                          <Flag className="h-4 w-4 mr-2" />
                          Report
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">From</p>
                        <p className="font-medium text-red-600">
                          {currentEmail.sender}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Subject</p>
                        <p className="font-medium">{currentEmail.subject}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Received</p>
                        <p className="font-medium">
                          {new Date(currentEmail.date).toLocaleString()}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Classification</p>
                        <Badge 
                          variant="destructive"
                          className="bg-red-100 text-red-800"
                        >
                          Fraud
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Analysis Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Classification</CardTitle>
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-red-600">Fraud</div>
                      <p className="text-xs text-muted-foreground">
                        Fraudulent email detected
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Suspicious Words</CardTitle>
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{suspiciousWords.length}</div>
                      <p className="text-xs text-muted-foreground">
                        High-risk keywords detected
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Authentication</CardTitle>
                      <Shield className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-red-600">Failed</div>
                      <p className="text-xs text-muted-foreground">
                        SPF, DKIM, DMARC all failed
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Main Analysis Content */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Risk Breakdown Pie Chart */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Classification Analysis</CardTitle>
                      <CardDescription>
                        Breakdown of email content indicators
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={classificationBreakdown}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              {classificationBreakdown.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex flex-col space-y-2 mt-4">
                        {classificationBreakdown.map((item, index) => (
                          <div key={index} className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <div 
                                className="w-3 h-3 rounded-full" 
                                style={{ backgroundColor: item.color }}
                              />
                              <span className="text-sm">{item.name}</span>
                            </div>
                            <span className="text-sm font-medium">{item.value}%</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Suspicious Words Analysis */}
                  <Card className="lg:col-span-2">
                    <CardHeader>
                      <CardTitle>Keywords Analysis</CardTitle>
                      <CardDescription>
                        Detected keywords that influenced classification
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 mb-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={wordAnalysisData}>
                            <XAxis dataKey="word" angle={-45} textAnchor="end" height={80} />
                            <YAxis />
                            <Tooltip 
                              content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                  const data = payload[0].payload;
                                  return (
                                    <div className="bg-white p-3 border rounded-lg shadow-lg">
                                      <p className="font-medium">{label}</p>
                                      <p className="text-sm">Count: {data.count}</p>
                                      <p className="text-sm">Risk Level: {suspiciousWords.find(w => w.word === label)?.riskLevel}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar 
                              dataKey="count" 
                              fill="#ef4444"
                              radius={[4, 4, 0, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      
                      <div className="space-y-2">
                        <h4 className="font-medium">Detected Keywords:</h4>
                        <div className="flex flex-wrap gap-2">
                          {suspiciousWords.map((word, index) => (
                            <Badge 
                              key={index}
                              variant={word.riskLevel === 'high' ? 'destructive' : 'secondary'}
                              className={word.riskLevel === 'high' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}
                            >
                              {word.word} ({word.count}x)
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Email Content Analysis */}
                <Card>
                  <CardHeader>
                    <CardTitle>Email Content with Threat Highlighting</CardTitle>
                    <CardDescription>
                      Original email content with suspicious elements highlighted
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gray-50 p-4 rounded-lg border">
                      <div className="space-y-3">
                        <div className="border-b pb-2">
                          <p className="text-sm text-muted-foreground">Subject:</p>
                          <p className="font-medium">{currentEmail.subject}</p>
                        </div>
                        <div 
                          className="whitespace-pre-wrap text-sm leading-relaxed"
                          dangerouslySetInnerHTML={{ 
                            __html: highlightSuspiciousWords(currentDetails.content) 
                          }}
                        />
                      </div>
                    </div>
                    
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-start space-x-2">
                        <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                        <div>
                          <p className="font-medium text-red-800">Security Alert</p>
                          <p className="text-sm text-red-700">
                            This email has been classified as fraud. Do not click any links, download attachments, 
                            or provide personal information.
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Technical Analysis */}
                <Card>
                  <CardHeader>
                    <CardTitle>Technical Analysis</CardTitle>
                    <CardDescription>
                      Email authentication and technical security checks
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      <div className="space-y-3">
                        <h4 className="font-medium">Authentication Results</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm">SPF Check</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.spfCheck}</Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm">DKIM Check</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.dkimCheck}</Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm">DMARC Check</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.dmarcCheck}</Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <h4 className="font-medium">Domain Analysis</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm">Domain Age</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.domainAge}</Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm">SSL Certificate</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.sslCertificate}</Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm">Reputation</span>
                            <Badge variant="destructive">{sampleEmail.technicalAnalysis.reputation}</Badge>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <h4 className="font-medium">Recommended Actions</h4>
                        <div className="space-y-2">
                          <Button variant="destructive" size="sm" className="w-full">
                            <AlertTriangle className="h-4 w-4 mr-2" />
                            Block Sender
                          </Button>
                          <Button variant="outline" size="sm" className="w-full">
                            <Flag className="h-4 w-4 mr-2" />
                            Report as Phishing
                          </Button>
                          <Button variant="secondary" size="sm" className="w-full">
                            <Shield className="h-4 w-4 mr-2" />
                            Add to Blacklist
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            );
          })()}
        </div>
      );
    } else {
      // For regular users, show Gmail integration
      return (
        <div className="p-6 space-y-6 max-w-7xl mx-auto">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Email Analysis</h1>
            <p className="text-muted-foreground">
              Connect your Gmail account and select an email to perform detailed security analysis
            </p>
          </div>
          
          <GmailIntegration onEmailSelect={handleEmailSelect} />
        </div>
      );
    }
  }

  // Use real email data or fallback to sample data for display
  const currentEmail = selectedEmail || sampleEmail;
  const currentDetails = emailDetails || {};

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">Email Analysis</h1>
          <p className="text-muted-foreground">
            Detailed security analysis of the selected email
          </p>
        </div>
        <Button 
          variant="outline" 
          onClick={() => setSelectedEmail(null)}
        >
          ← Back to Email List
        </Button>
      </div>

      {/* Email Overview */}
      <Card className={`border-l-4 ${currentEmail.riskLevel === 'high' ? 'border-l-red-500' : currentEmail.riskLevel === 'medium' ? 'border-l-yellow-500' : 'border-l-green-500'}`}>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <CardTitle className={`text-xl ${currentEmail.riskLevel === 'high' ? 'text-red-700' : currentEmail.riskLevel === 'medium' ? 'text-yellow-700' : 'text-green-700'}`}>
                {currentEmail.riskLevel === 'high' ? 'Fraud Email Detected' : 
                 currentEmail.riskLevel === 'medium' ? 'Spam Email' : 
                 'Normal Email'}
              </CardTitle>
              <CardDescription>
                {currentEmail.riskLevel === 'high' ? 'This email has been classified as fraudulent' :
                 currentEmail.riskLevel === 'medium' ? 'This email has been classified as spam' :
                 'This email has been classified as normal'}
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Badge 
                variant={currentEmail.riskLevel === 'high' ? 'destructive' : currentEmail.riskLevel === 'medium' ? 'secondary' : 'outline'} 
                className={`text-lg px-3 py-1 ${
                  currentEmail.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                  currentEmail.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}
              >
                Confidence: {currentEmail.riskScore || 0}%
              </Badge>
              <Button variant="outline" size="sm">
                <Flag className="h-4 w-4 mr-2" />
                Report
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">From</p>
              <p className={`font-medium ${currentEmail.riskLevel === 'high' ? 'text-red-600' : ''}`}>
                {currentEmail.sender}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Subject</p>
              <p className="font-medium">{currentEmail.subject || '(No Subject)'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Received</p>
              <p className="font-medium">
                {new Date(currentEmail.date || currentEmail.receivedDate).toLocaleString()}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Classification</p>
              <Badge 
                variant={currentEmail.riskLevel === 'high' ? 'destructive' : currentEmail.riskLevel === 'medium' ? 'secondary' : 'outline'}
                className={
                  currentEmail.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                  currentEmail.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }
              >
                {currentEmail.riskLevel === 'high' ? 'Fraud' :
                 currentEmail.riskLevel === 'medium' ? 'Spam' :
                 'Normal'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Classification</CardTitle>
            <AlertTriangle className={`h-4 w-4 ${currentEmail.riskLevel === 'high' ? 'text-red-500' : currentEmail.riskLevel === 'medium' ? 'text-yellow-500' : 'text-green-500'}`} />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${currentEmail.riskLevel === 'high' ? 'text-red-600' : currentEmail.riskLevel === 'medium' ? 'text-yellow-600' : 'text-green-600'}`}>
              {currentEmail.riskLevel === 'high' ? 'Fraud' : currentEmail.riskLevel === 'medium' ? 'Spam' : 'Normal'}
            </div>
            <p className="text-xs text-muted-foreground">
              {currentEmail.riskLevel === 'high' ? 'Fraudulent email detected' : 
               currentEmail.riskLevel === 'medium' ? 'Spam email detected' : 
               'Normal email'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Suspicious Words</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentDetails.matchedKeywords || suspiciousWords.length}</div>
            <p className="text-xs text-muted-foreground">
              {selectedEmail ? 'Keywords detected in this email' : 'High-risk keywords detected'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Authentication</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${currentDetails.headers?.['Authentication-Results']?.includes('fail') ? 'text-red-600' : 'text-green-600'}`}>
              {currentDetails.headers?.['Authentication-Results']?.includes('fail') ? 'Failed' : selectedEmail ? 'Passed' : 'Failed'}
            </div>
            <p className="text-xs text-muted-foreground">
              {selectedEmail ? 'Live authentication check' : 'SPF, DKIM, DMARC all failed'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analysis Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Breakdown Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Classification Analysis</CardTitle>
            <CardDescription>
              Breakdown of email content indicators
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={classificationBreakdown}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {classificationBreakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col space-y-2 mt-4">
              {classificationBreakdown.map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm">{item.name}</span>
                  </div>
                  <span className="text-sm font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Suspicious Words Analysis */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Keywords Analysis</CardTitle>
            <CardDescription>
              Detected keywords that influenced classification
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={wordAnalysisData}>
                  <XAxis dataKey="word" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip 
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 border rounded-lg shadow-lg">
                            <p className="font-medium">{label}</p>
                            <p className="text-sm">Count: {data.count}</p>
                            <p className="text-sm">Risk Level: {suspiciousWords.find(w => w.word === label)?.riskLevel}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar 
                    dataKey="count" 
                    fill="#ef4444"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-2">
              <h4 className="font-medium">Detected Keywords:</h4>
              <div className="flex flex-wrap gap-2">
                {suspiciousWords.map((word, index) => (
                  <Badge 
                    key={index}
                    variant={word.riskLevel === 'high' ? 'destructive' : 'secondary'}
                    className={word.riskLevel === 'high' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}
                  >
                    {word.word} ({word.count}x)
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Email Content Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Email Content with Threat Highlighting</CardTitle>
          <CardDescription>
            Original email content with suspicious elements highlighted
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-50 p-4 rounded-lg border">
            <div className="space-y-3">
              <div className="border-b pb-2">
                <p className="text-sm text-muted-foreground">Subject:</p>
                <p className="font-medium">{currentEmail.subject || '(No Subject)'}</p>
              </div>
              {currentDetails.content ? (
                <div 
                  className="whitespace-pre-wrap text-sm leading-relaxed"
                  dangerouslySetInnerHTML={{ 
                    __html: highlightSuspiciousWords(currentDetails.content) 
                  }}
                />
              ) : (
                <div className="text-sm text-muted-foreground">
                  <p>Email content preview:</p>
                  <p className="mt-2 italic">{currentEmail.snippet || 'No content preview available'}</p>
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
              <div>
                <p className="font-medium text-red-800">Security Alert</p>
                <p className="text-sm text-red-700">
                  This email has been classified as fraud. Do not click any links, download attachments, 
                  or provide personal information.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Analysis</CardTitle>
          <CardDescription>
            Email authentication and technical security checks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="space-y-3">
              <h4 className="font-medium">Authentication Results</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm">SPF Check</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.spfCheck}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">DKIM Check</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.dkimCheck}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">DMARC Check</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.dmarcCheck}</Badge>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium">Domain Analysis</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Domain Age</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.domainAge}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">SSL Certificate</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.sslCertificate}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">Reputation</span>
                  <Badge variant="destructive">{sampleEmail.technicalAnalysis.reputation}</Badge>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium">Recommended Actions</h4>
              <div className="space-y-2">
                <Button variant="destructive" size="sm" className="w-full">
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  Block Sender
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  <Flag className="h-4 w-4 mr-2" />
                  Report as Phishing
                </Button>
                <Button variant="secondary" size="sm" className="w-full">
                  <Shield className="h-4 w-4 mr-2" />
                  Add to Blacklist
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}