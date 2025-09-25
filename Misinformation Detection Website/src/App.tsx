import { useState, useEffect } from 'react';
import { Navigation } from './components/navigation';
import { Dashboard } from './components/dashboard';
import { TrendAnalysis } from './components/trend-analysis';
import { RiskAnalysis } from './components/risk-analysis';
import { EmailAnalysis } from './components/email-analysis';
import { AuthPage } from './components/auth-page';
import { GmailCallback } from './components/gmail-callback';

type Page = 'dashboard' | 'trends' | 'risks' | 'email-analysis' | 'auth';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('auth');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isGmailConnected, setIsGmailConnected] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const handleLogin = (demoMode = false) => {
    setIsAuthenticated(true);
    setIsDemoMode(demoMode);
    if (demoMode) {
      // Demo users bypass Gmail connection requirement
      setCurrentPage('dashboard');
    }
    // For regular users, don't automatically redirect to dashboard - wait for Gmail connection
  };

  const handleGmailConnected = () => {
    setIsGmailConnected(true);
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setIsGmailConnected(false);
    setIsDemoMode(false);
    setCurrentPage('auth');
  };

  // Check Gmail connection status on app load
  useEffect(() => {
    const checkGmailConnection = async () => {
      if (isAuthenticated) {
        try {
          const { projectId, publicAnonKey } = await import('./utils/supabase/info');
          const response = await fetch(`https://${projectId}.supabase.co/functions/v1/make-server-4ca42d89/gmail/status`, {
            headers: {
              'Authorization': `Bearer ${publicAnonKey}`,
            },
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.connected) {
              setIsGmailConnected(true);
              setCurrentPage('dashboard');
            }
          }
        } catch (error) {
          console.error('Failed to check Gmail connection:', error);
        }
      }
    };

    checkGmailConnection();
  }, [isAuthenticated]);

  // Handle Gmail OAuth callback
  if (window.location.search.includes('code=') || window.location.search.includes('error=')) {
    return <GmailCallback onComplete={handleGmailConnected} />;
  }

  // Show auth page if not authenticated or (Gmail not connected and not in demo mode)
  if (!isAuthenticated || (!isGmailConnected && !isDemoMode)) {
    return <AuthPage onLogin={handleLogin} isAuthenticated={isAuthenticated} />;
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'trends':
        return <TrendAnalysis />;
      case 'risks':
        return <RiskAnalysis />;
      case 'email-analysis':
        return <EmailAnalysis isDemoMode={isDemoMode} />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation 
        currentPage={currentPage} 
        onNavigate={setCurrentPage}
        onLogout={handleLogout}
      />
      <main className="pt-16">
        {renderPage()}
      </main>
    </div>
  );
}