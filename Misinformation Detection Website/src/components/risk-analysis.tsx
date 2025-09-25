import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Avatar, AvatarFallback } from './ui/avatar';
import { AlertTriangle, Shield, Search, Filter, Mail, Clock, Users } from 'lucide-react';
import { useState } from 'react';

const fraudSenders = [
  {
    email: 'urgent.support@fake-bank.com',
    domain: 'fake-bank.com',
    classification: 'Fraud',
    confidence: 95,
    lastSeen: '2024-09-23',
    emailCount: 12,
    blocked: true
  },
  {
    email: 'winner@lottery-scam.net',
    domain: 'lottery-scam.net',
    classification: 'Fraud',
    confidence: 92,
    lastSeen: '2024-09-22',
    emailCount: 8,
    blocked: true
  },
  {
    email: 'love.connection@romance-fraud.org',
    domain: 'romance-fraud.org',
    classification: 'Fraud',
    confidence: 89,
    lastSeen: '2024-09-21',
    emailCount: 15,
    blocked: true
  },
  {
    email: 'tech.help@fake-microsoft.com',
    domain: 'fake-microsoft.com',
    classification: 'Fraud',
    confidence: 87,
    lastSeen: '2024-09-20',
    emailCount: 6,
    blocked: false
  }
];

const spamSenders = [
  {
    email: 'promotions@suspicious-deals.co',
    domain: 'suspicious-deals.co',
    classification: 'Spam',
    confidence: 78,
    lastSeen: '2024-09-23',
    emailCount: 24,
    blocked: false
  },
  {
    email: 'newsletter@questionable-source.info',
    domain: 'questionable-source.info',
    classification: 'Spam',
    confidence: 72,
    lastSeen: '2024-09-22',
    emailCount: 18,
    blocked: false
  },
  {
    email: 'alerts@crypto-pump.biz',
    domain: 'crypto-pump.biz',
    classification: 'Spam',
    confidence: 85,
    lastSeen: '2024-09-23',
    emailCount: 9,
    blocked: false
  },
  {
    email: 'offers@flash-sale.club',
    domain: 'flash-sale.club',
    classification: 'Spam',
    confidence: 69,
    lastSeen: '2024-09-21',
    emailCount: 31,
    blocked: false
  }
];

const normalSenders = [
  {
    email: 'updates@legitimate-service.com',
    domain: 'legitimate-service.com',
    classification: 'Normal',
    confidence: 92,
    lastSeen: '2024-09-23',
    emailCount: 45,
    blocked: false
  },
  {
    email: 'support@trusted-company.org',
    domain: 'trusted-company.org',
    classification: 'Normal',
    confidence: 96,
    lastSeen: '2024-09-23',
    emailCount: 32,
    blocked: false
  },
  {
    email: 'info@verified-business.net',
    domain: 'verified-business.net',
    classification: 'Normal',
    confidence: 88,
    lastSeen: '2024-09-22',
    emailCount: 28,
    blocked: false
  },
  {
    email: 'news@reputable-source.edu',
    domain: 'reputable-source.edu',
    classification: 'Normal',
    confidence: 94,
    lastSeen: '2024-09-23',
    emailCount: 56,
    blocked: false
  }
];

function SenderCard({ sender, classification }: { sender: any, classification: string }) {
  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'Fraud':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'Spam':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Normal':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getClassificationBadgeColor = (classification: string) => {
    switch (classification) {
      case 'Fraud':
        return 'bg-red-100 text-red-800';
      case 'Spam':
        return 'bg-orange-100 text-orange-800';
      case 'Normal':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getBorderColor = () => {
    switch (classification) {
      case 'Fraud':
        return 'border-l-red-500';
      case 'Spam':
        return 'border-l-orange-500';
      case 'Normal':
        return 'border-l-green-500';
      default:
        return 'border-l-gray-500';
    }
  };

  return (
    <Card className={`border-l-4 ${getBorderColor()}`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between space-x-4">
          <div className="flex items-start space-x-3 flex-1 min-w-0">
            <Avatar className="h-10 w-10">
              <AvatarFallback className={getClassificationColor(sender.classification)}>
                {sender.email.charAt(0).toUpperCase()}
              </AvatarFallback>
            </Avatar>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2 mb-1">
                <p className="font-medium truncate">{sender.email}</p>
                {sender.blocked && <Badge variant="destructive" className="text-xs">Blocked</Badge>}
              </div>
              
              <p className="text-sm text-muted-foreground mb-2">{sender.domain}</p>
              
              <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                <div className="flex items-center space-x-1">
                  <Mail className="h-3 w-3" />
                  <span>{sender.emailCount} emails</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Clock className="h-3 w-3" />
                  <span>Last: {new Date(sender.lastSeen).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col items-end space-y-2">
            <div className={`px-2 py-1 rounded text-xs font-medium ${getClassificationColor(sender.classification)}`}>
              Confidence: {sender.confidence}%
            </div>
            <Badge 
              variant="outline" 
              className={getClassificationBadgeColor(sender.classification)}
            >
              {sender.classification}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function RiskAnalysis() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBy, setFilterBy] = useState('all');

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Sender Classification Analysis</h1>
        <p className="text-muted-foreground">
          Analysis of email senders classified as Fraud, Spam, or Normal using AI detection technology
        </p>
      </div>

      {/* Classification Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fraud Senders</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{fraudSenders.length}</div>
            <p className="text-xs text-muted-foreground">Dangerous fraud attempts</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Spam Senders</CardTitle>
            <Shield className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">{spamSenders.length}</div>
            <p className="text-xs text-muted-foreground">Unwanted promotional emails</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Normal Senders</CardTitle>
            <Users className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{normalSenders.length}</div>
            <p className="text-xs text-muted-foreground">Legitimate communications</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Blocked Senders</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {[...fraudSenders, ...spamSenders].filter(s => s.blocked).length}
            </div>
            <p className="text-xs text-muted-foreground">Auto-blocked threats</p>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filter */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search senders by email or domain..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select value={filterBy} onValueChange={setFilterBy}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Classifications</SelectItem>
                <SelectItem value="fraud">Fraud Only</SelectItem>
                <SelectItem value="spam">Spam Only</SelectItem>
                <SelectItem value="normal">Normal Only</SelectItem>
                <SelectItem value="blocked">Blocked Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Fraud Senders */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="h-5 w-5 text-red-500" />
          <h2 className="text-xl font-semibold text-red-700">Fraud Senders</h2>
          <Badge variant="destructive">{fraudSenders.length}</Badge>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {fraudSenders.map((sender, index) => (
            <SenderCard key={index} sender={sender} classification="Fraud" />
          ))}
        </div>
      </div>

      {/* Spam Senders */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Shield className="h-5 w-5 text-orange-500" />
          <h2 className="text-xl font-semibold text-orange-700">Spam Senders</h2>
          <Badge variant="secondary" className="bg-orange-100 text-orange-800">{spamSenders.length}</Badge>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {spamSenders.map((sender, index) => (
            <SenderCard key={index} sender={sender} classification="Spam" />
          ))}
        </div>
      </div>

      {/* Normal Senders */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Users className="h-5 w-5 text-green-500" />
          <h2 className="text-xl font-semibold text-green-700">Normal Senders</h2>
          <Badge variant="outline" className="bg-green-100 text-green-800">{normalSenders.length}</Badge>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {normalSenders.map((sender, index) => (
            <SenderCard key={index} sender={sender} classification="Normal" />
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Manage sender classifications and security settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button variant="destructive">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Block All Fraud
            </Button>
            <Button variant="outline">
              <Shield className="h-4 w-4 mr-2" />
              Review Spam Senders
            </Button>
            <Button variant="secondary">
              <Users className="h-4 w-4 mr-2" />
              Export Sender List
            </Button>
            <Button variant="outline">
              <Filter className="h-4 w-4 mr-2" />
              Configure Filters
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}