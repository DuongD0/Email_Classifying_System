import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Shield, TrendingUp, Users, Mail, LogOut, BarChart3, CheckCircle } from 'lucide-react';

type Page = 'dashboard' | 'trends' | 'risks' | 'email-analysis' | 'auth';

interface NavigationProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function Navigation({ currentPage, onNavigate, onLogout }: NavigationProps) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'trends', label: 'Email Trends', icon: TrendingUp },
    { id: 'risks', label: 'Risk Analysis', icon: Users },
    { id: 'email-analysis', label: 'Email Analysis', icon: Mail },
  ] as const;

  return (
    <nav className="fixed top-0 w-full bg-background border-b border-border z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="h-8 w-8 text-primary" />
                <span className="text-xl font-semibold">EmailGuard</span>
              </div>
              <Badge variant="outline" className="bg-green-100 text-green-800 border-green-300 hidden sm:flex">
                <CheckCircle className="h-3 w-3 mr-1" />
                Gmail Connected
              </Badge>
            </div>
            
            <div className="hidden md:flex space-x-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.id}
                    variant={currentPage === item.id ? 'default' : 'ghost'}
                    onClick={() => onNavigate(item.id as Page)}
                    className="flex items-center space-x-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </Button>
                );
              })}
            </div>
          </div>

          <Button
            variant="outline"
            onClick={onLogout}
            className="flex items-center space-x-2"
          >
            <LogOut className="h-4 w-4" />
            <span>Logout</span>
          </Button>
        </div>
      </div>
      
      {/* Mobile navigation */}
      <div className="md:hidden border-t border-border">
        <div className="flex overflow-x-auto px-4 py-2 space-x-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Button
                key={item.id}
                variant={currentPage === item.id ? 'default' : 'ghost'}
                onClick={() => onNavigate(item.id as Page)}
                size="sm"
                className="flex items-center space-x-1 whitespace-nowrap"
              >
                <Icon className="h-3 w-3" />
                <span className="text-xs">{item.label}</span>
              </Button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}