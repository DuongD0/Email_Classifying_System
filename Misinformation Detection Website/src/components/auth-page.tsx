import { useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { Alert, AlertDescription } from './ui/alert';
import { Shield, Mail, Github, Chrome, Facebook, Twitter, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';
import { projectId, publicAnonKey } from '../utils/supabase/info';

interface AuthPageProps {
  onLogin: (demoMode?: boolean) => void;
  isAuthenticated?: boolean;
}

export function AuthPage({ onLogin, isAuthenticated = false }: AuthPageProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulate authentication
    onLogin();
  };

  const handleSocialAuth = (provider: string) => {
    console.log(`Authenticating with ${provider}`);
    // Simulate social authentication
    onLogin();
  };

  const connectGmail = async () => {
    try {
      setLoading(true);
      setError(null);
      
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

  // If authenticated but Gmail not connected, show Gmail connection screen
  if (isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center">
            <div className="flex justify-center items-center space-x-2 mb-4">
              <Shield className="h-12 w-12 text-blue-600" />
              <span className="text-3xl font-bold text-gray-900">EmailGuard</span>
            </div>
            <p className="text-gray-600">
              Connect your Gmail account to start protecting your inbox
            </p>
          </div>

          <Card className="shadow-lg">
            <CardHeader className="space-y-1">
              <CardTitle className="text-2xl text-center">
                Gmail Connection Required
              </CardTitle>
              <CardDescription className="text-center">
                EmailGuard needs access to your Gmail to analyze your emails for threats
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Alert>
                <Mail className="h-4 w-4" />
                <AlertDescription>
                  We only read email content for security analysis. Your emails are never stored or shared.
                </AlertDescription>
              </Alert>

              <div className="space-y-4">
                <h4 className="font-medium">What EmailGuard analyzes:</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Suspicious keywords and phishing patterns</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Sender authentication (SPF, DKIM, DMARC)</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Domain reputation and age verification</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Malicious links and attachment scanning</span>
                  </li>
                </ul>
              </div>

              <Button 
                onClick={connectGmail} 
                disabled={loading}
                className="w-full"
                size="lg"
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

              <div className="text-center text-xs text-muted-foreground">
                <p>
                  By connecting Gmail, you agree to allow EmailGuard read-only access to your emails for security analysis only.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo and Title */}
        <div className="text-center">
          <div className="flex justify-center items-center space-x-2 mb-4">
            <Shield className="h-12 w-12 text-blue-600" />
            <span className="text-3xl font-bold text-gray-900">EmailGuard</span>
          </div>
          <p className="text-gray-600">
            Advanced email misinformation detection powered by AI
          </p>
        </div>

        <Card className="shadow-lg">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center">
              {isLogin ? 'Welcome back' : 'Create account'}
            </CardTitle>
            <CardDescription className="text-center">
              {isLogin 
                ? 'Sign in and connect Gmail to start analyzing' 
                : 'Sign up and connect Gmail to protect your inbox'
              }
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Social Login Options */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                onClick={() => handleSocialAuth('google')}
                className="flex items-center justify-center space-x-2"
              >
                <Chrome className="h-4 w-4" />
                <span>Google</span>
              </Button>
              <Button
                variant="outline"
                onClick={() => handleSocialAuth('github')}
                className="flex items-center justify-center space-x-2"
              >
                <Github className="h-4 w-4" />
                <span>GitHub</span>
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                onClick={() => handleSocialAuth('facebook')}
                className="flex items-center justify-center space-x-2"
              >
                <Facebook className="h-4 w-4" />
                <span>Facebook</span>
              </Button>
              <Button
                variant="outline"
                onClick={() => handleSocialAuth('twitter')}
                className="flex items-center justify-center space-x-2"
              >
                <Twitter className="h-4 w-4" />
                <span>Twitter</span>
              </Button>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <Separator />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Or continue with
                </span>
              </div>
            </div>

            {/* Email/Password Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
              
              {isLogin && (
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <input type="checkbox" id="remember" className="rounded" />
                    <Label htmlFor="remember" className="text-sm">
                      Remember me
                    </Label>
                  </div>
                  <Button variant="link" className="px-0 text-sm">
                    Forgot password?
                  </Button>
                </div>
              )}

              <Button type="submit" className="w-full">
                {isLogin ? 'Sign In' : 'Sign Up'}
              </Button>
            </form>

            <div className="text-center">
              <Button
                variant="link"
                onClick={() => setIsLogin(!isLogin)}
                className="text-sm"
              >
                {isLogin 
                  ? "Don't have an account? Sign up" 
                  : 'Already have an account? Sign in'
                }
              </Button>
            </div>

            {/* Gmail Requirement Notice */}
            <Alert>
              <Mail className="h-4 w-4" />
              <AlertDescription>
                After signing in, you'll need to connect your Gmail account to use EmailGuard's protection features.
              </AlertDescription>
            </Alert>

            {/* Demo Button */}
            <div className="pt-4 border-t">
              <Button
                variant="secondary"
                onClick={() => onLogin(true)}
                className="w-full"
              >
                Continue as Demo User
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="text-center text-sm text-gray-500">
          <p>
            By signing in, you agree to our{' '}
            <Button variant="link" className="p-0 h-auto text-sm">
              Terms of Service
            </Button>
            {' '}and{' '}
            <Button variant="link" className="p-0 h-auto text-sm">
              Privacy Policy
            </Button>
          </p>
        </div>
      </div>
    </div>
  );
}