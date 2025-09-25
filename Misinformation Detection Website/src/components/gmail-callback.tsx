import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Alert, AlertDescription } from './ui/alert';
import { CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { projectId, publicAnonKey } from '../utils/supabase/info';

interface GmailCallbackProps {
  onComplete: () => void;
}

export function GmailCallback({ onComplete }: GmailCallbackProps) {
  const [status, setStatus] = useState<'processing' | 'success' | 'error'>('processing');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');
        const error = urlParams.get('error');

        if (error) {
          throw new Error(`Authentication failed: ${error}`);
        }

        if (!code) {
          throw new Error('No authorization code received');
        }

        const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/callback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${publicAnonKey}`,
          },
          body: JSON.stringify({ code }),
        });

        if (response.ok) {
          setStatus('success');
          // Call the completion callback after a short delay
          setTimeout(() => {
            onComplete();
          }, 2000);
        } else {
          const data = await response.json();
          throw new Error(data.error || 'Failed to complete authentication');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
        setStatus('error');
      }
    };

    handleCallback();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="flex items-center justify-center space-x-2">
            {status === 'processing' && <RefreshCw className="h-5 w-5 animate-spin" />}
            {status === 'success' && <CheckCircle className="h-5 w-5 text-green-500" />}
            {status === 'error' && <AlertCircle className="h-5 w-5 text-red-500" />}
            <span>Gmail Authentication</span>
          </CardTitle>
          <CardDescription>
            {status === 'processing' && 'Processing your Gmail authentication...'}
            {status === 'success' && 'Successfully connected to Gmail!'}
            {status === 'error' && 'Authentication failed'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {status === 'processing' && (
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                Please wait while we complete the authentication process...
              </p>
            </div>
          )}

          {status === 'success' && (
            <div className="space-y-4">
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Your Gmail account has been successfully connected to EmailGuard. 
                  You can now analyze your emails for security threats.
                </AlertDescription>
              </Alert>
              <p className="text-sm text-center text-muted-foreground">
                Redirecting to the main application...
              </p>
            </div>
          )}

          {status === 'error' && (
            <div className="space-y-4">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
              <Button 
                onClick={() => onComplete()} 
                className="w-full"
              >
                Return to App
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}