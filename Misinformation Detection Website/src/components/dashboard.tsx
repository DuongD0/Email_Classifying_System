import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Shield, Mail, TrendingUp, Clock, Users, CheckCircle } from 'lucide-react';

const overallAnalysisData = [
  { name: 'Normal', value: 68, color: '#22c55e' },
  { name: 'Spam', value: 22, color: '#f59e0b' },
  { name: 'Fraud', value: 10, color: '#ef4444' }
];

const buzzwordsData = [
  { word: 'urgent', count: 45, risk: 'high' },
  { word: 'limited time', count: 38, risk: 'high' },
  { word: 'click here', count: 32, risk: 'medium' },
  { word: 'free money', count: 28, risk: 'high' },
  { word: 'act now', count: 25, risk: 'high' },
  { word: 'congratulations', count: 22, risk: 'medium' },
  { word: 'verify account', count: 18, risk: 'medium' },
  { word: 'suspended', count: 15, risk: 'high' }
];

const weeklyActivityData = [
  { day: 'Mon', normal: 120, spam: 35, fraud: 8 },
  { day: 'Tue', normal: 98, spam: 28, fraud: 12 },
  { day: 'Wed', normal: 105, spam: 42, fraud: 6 },
  { day: 'Thu', normal: 134, spam: 31, fraud: 9 },
  { day: 'Fri', normal: 156, spam: 45, fraud: 15 },
  { day: 'Sat', normal: 89, spam: 22, fraud: 4 },
  { day: 'Sun', normal: 76, spam: 18, fraud: 3 }
];

const COLORS = ['#22c55e', '#f59e0b', '#ef4444'];

export function Dashboard() {
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="space-y-4">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">Email Security Dashboard</h1>
          <p className="text-muted-foreground">
            Comprehensive analysis of your email security and misinformation detection
          </p>
        </div>

        {/* Gmail Connection Status */}
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Gmail Connected:</strong> EmailGuard is actively monitoring and analyzing your emails for security threats.
          </AlertDescription>
        </Alert>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Emails Analyzed</CardTitle>
            <Mail className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2,847</div>
            <p className="text-xs text-muted-foreground">+12% from last week</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fraud & Spam Detected</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">287</div>
            <p className="text-xs text-muted-foreground">-8% from last week</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Protection Rate</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">89.9%</div>
            <p className="text-xs text-muted-foreground">+2.1% from last week</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0.3s</div>
            <p className="text-xs text-muted-foreground">Real-time analysis</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Analysis Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Overall Email Analysis Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Overall Email Classification</CardTitle>
            <CardDescription>
              Distribution of email types detected in the past 30 days
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={overallAnalysisData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {overallAnalysisData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center space-x-4 mt-4">
              {overallAnalysisData.map((item, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-muted-foreground">{item.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Weekly Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Weekly Email Classification</CardTitle>
            <CardDescription>
              Email classification results by day of the week
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weeklyActivityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="normal" stackId="a" fill="#22c55e" name="Normal" />
                  <Bar dataKey="spam" stackId="a" fill="#f59e0b" name="Spam" />
                  <Bar dataKey="fraud" stackId="a" fill="#ef4444" name="Fraud" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Buzzwords Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Suspicious Keywords Analysis</CardTitle>
          <CardDescription>
            Most frequently detected suspicious words and phrases in analyzed emails
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {buzzwordsData.map((buzzword, index) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="space-y-1">
                  <p className="font-medium">{buzzword.word}</p>
                  <p className="text-sm text-muted-foreground">{buzzword.count} occurrences</p>
                </div>
                <Badge 
                  variant={buzzword.risk === 'high' ? 'destructive' : 'secondary'}
                  className={buzzword.risk === 'high' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}
                >
                  {buzzword.risk}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Analysis Summary</CardTitle>
          <CardDescription>
            Latest email security alerts and analysis results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4 p-3 border rounded-lg">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              <div className="flex-1">
                <p className="font-medium">Fraud email detected</p>
                <p className="text-sm text-muted-foreground">Sender: fake.bank@suspicious-domain.com</p>
              </div>
              <Badge variant="destructive">Fraud</Badge>
            </div>
            
            <div className="flex items-center space-x-4 p-3 border rounded-lg">
              <TrendingUp className="h-5 w-5 text-yellow-500" />
              <div className="flex-1">
                <p className="font-medium">Unusual spike in spam emails</p>
                <p className="text-sm text-muted-foreground">45% increase compared to last week</p>
              </div>
              <Badge variant="secondary">Spam</Badge>
            </div>
            
            <div className="flex items-center space-x-4 p-3 border rounded-lg">
              <Shield className="h-5 w-5 text-green-500" />
              <div className="flex-1">
                <p className="font-medium">System performance optimal</p>
                <p className="text-sm text-muted-foreground">All email filters operating normally</p>
              </div>
              <Badge variant="outline" className="bg-green-100 text-green-800">Normal</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}