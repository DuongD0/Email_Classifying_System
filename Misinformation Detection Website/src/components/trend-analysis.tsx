import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, Calendar, Filter } from 'lucide-react';
import { useState } from 'react';

const monthlyTrendData = [
  { month: 'Jan', scamEmails: 45, totalEmails: 892, percentage: 5.0 },
  { month: 'Feb', scamEmails: 52, totalEmails: 934, percentage: 5.6 },
  { month: 'Mar', scamEmails: 38, totalEmails: 876, percentage: 4.3 },
  { month: 'Apr', scamEmails: 67, totalEmails: 1024, percentage: 6.5 },
  { month: 'May', scamEmails: 74, totalEmails: 1156, percentage: 6.4 },
  { month: 'Jun', scamEmails: 89, totalEmails: 1298, percentage: 6.9 },
  { month: 'Jul', scamEmails: 95, totalEmails: 1387, percentage: 6.8 },
  { month: 'Aug', scamEmails: 112, totalEmails: 1456, percentage: 7.7 },
  { month: 'Sep', scamEmails: 98, totalEmails: 1342, percentage: 7.3 }
];

const dailyTrendData = [
  { date: '2024-09-15', scamEmails: 12, totalEmails: 156 },
  { date: '2024-09-16', scamEmails: 8, totalEmails: 134 },
  { date: '2024-09-17', scamEmails: 15, totalEmails: 167 },
  { date: '2024-09-18', scamEmails: 18, totalEmails: 189 },
  { date: '2024-09-19', scamEmails: 22, totalEmails: 198 },
  { date: '2024-09-20', scamEmails: 16, totalEmails: 145 },
  { date: '2024-09-21', scamEmails: 9, totalEmails: 112 },
  { date: '2024-09-22', scamEmails: 11, totalEmails: 128 },
  { date: '2024-09-23', scamEmails: 14, totalEmails: 156 }
];

const emailTypeData = [
  { type: 'Normal', count: 1954, trend: '+3%', color: '#22c55e' },
  { type: 'Spam', count: 634, trend: '+8%', color: '#f59e0b' },
  { type: 'Fraud', count: 287, trend: '+12%', color: '#ef4444' }
];

const timePatternData = [
  { hour: '0', count: 2 },
  { hour: '1', count: 1 },
  { hour: '2', count: 0 },
  { hour: '3', count: 1 },
  { hour: '4', count: 3 },
  { hour: '5', count: 5 },
  { hour: '6', count: 8 },
  { hour: '7', count: 12 },
  { hour: '8', count: 18 },
  { hour: '9', count: 25 },
  { hour: '10', count: 32 },
  { hour: '11', count: 28 },
  { hour: '12', count: 22 },
  { hour: '13', count: 26 },
  { hour: '14', count: 31 },
  { hour: '15', count: 29 },
  { hour: '16', count: 24 },
  { hour: '17', count: 19 },
  { hour: '18', count: 15 },
  { hour: '19', count: 12 },
  { hour: '20', count: 8 },
  { hour: '21', count: 6 },
  { hour: '22', count: 4 },
  { hour: '23', count: 3 }
];

export function TrendAnalysis() {
  const [timeRange, setTimeRange] = useState('monthly');

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="flex justify-between items-start">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">Scam Email Trends</h1>
          <p className="text-muted-foreground">
            Analyze patterns and trends in malicious email detection over time
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="daily">Daily</SelectItem>
              <SelectItem value="weekly">Weekly</SelectItem>
              <SelectItem value="monthly">Monthly</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Key Trend Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">This Month's Scams</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">287</div>
            <div className="flex items-center space-x-2 text-xs">
              <TrendingUp className="h-3 w-3 text-red-500" />
              <span className="text-red-500">+12% from last month</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Detection Rate</CardTitle>
            <Filter className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">94.2%</div>
            <div className="flex items-center space-x-2 text-xs">
              <TrendingUp className="h-3 w-3 text-green-500" />
              <span className="text-green-500">+2.1% improvement</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Daily</CardTitle>
            <Calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">9.3</div>
            <div className="flex items-center space-x-2 text-xs">
              <TrendingDown className="h-3 w-3 text-green-500" />
              <span className="text-green-500">-5% from last week</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Trend Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Scam Email Detection Trends</CardTitle>
          <CardDescription>
            Monthly breakdown of detected malicious emails sent to your inbox
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlyTrendData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis yAxisId="left" orientation="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white p-3 border rounded-lg shadow-lg">
                          <p className="font-medium">{`${label} 2024`}</p>
                          <p className="text-red-600">{`Scam Emails: ${payload[0]?.value}`}</p>
                          <p className="text-blue-600">{`Total Emails: ${payload[1]?.value}`}</p>
                          <p className="text-gray-600">{`Percentage: ${payload[0]?.payload?.percentage}%`}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar yAxisId="left" dataKey="scamEmails" fill="#ef4444" name="Scam Emails" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="totalEmails" fill="#3b82f6" name="Total Emails" radius={[4, 4, 0, 0]} opacity={0.6} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Daily Trend and Hourly Patterns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Daily Activity</CardTitle>
            <CardDescription>
              Scam email detection in the last 9 days
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dailyTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                      weekday: 'long', 
                      month: 'short', 
                      day: 'numeric' 
                    })}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="scamEmails" 
                    stroke="#ef4444" 
                    fill="#ef4444" 
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Hourly Distribution</CardTitle>
            <CardDescription>
              When scam emails are most commonly received
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={timePatternData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => `${value}:00 - ${parseInt(value) + 1}:00`}
                  />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Email Classification Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Email Classification Analysis</CardTitle>
          <CardDescription>
            Breakdown of email types detected by EmailGuard's classification system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {emailTypeData.map((emailType, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: emailType.color }}
                    />
                    <p className="font-medium">{emailType.type}</p>
                  </div>
                  <p className="text-2xl font-bold">{emailType.count}</p>
                  <div className="flex items-center space-x-1">
                    {emailType.trend.startsWith('+') ? (
                      <TrendingUp className={`h-3 w-3 ${emailType.type === 'Normal' ? 'text-green-500' : 'text-red-500'}`} />
                    ) : (
                      <TrendingDown className="h-3 w-3 text-green-500" />
                    )}
                    <span className={`text-xs ${emailType.trend.startsWith('+') && emailType.type !== 'Normal' ? 'text-red-500' : 'text-green-500'}`}>
                      {emailType.trend}
                    </span>
                  </div>
                </div>
                <Badge 
                  variant="outline" 
                  style={{ borderColor: emailType.color, color: emailType.color }}
                >
                  {((emailType.count / 2875) * 100).toFixed(1)}%
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}