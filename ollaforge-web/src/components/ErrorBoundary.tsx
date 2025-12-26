/**
 * ErrorBoundary component for catching and displaying React errors.
 *
 * Features:
 * - Catches JavaScript errors in child component tree
 * - Displays fallback UI with error details
 * - Provides retry functionality
 *
 * Requirements: 1.4, 3.4
 */

import { Component, type ReactNode, type ErrorInfo } from 'react';
import { Result, Button, Typography, Collapse } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

const { Paragraph, Text } = Typography;

interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Fallback component to render on error */
  fallback?: ReactNode;
  /** Callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Custom error title */
  title?: string;
  /** Custom error description */
  description?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * ErrorBoundary catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    
    // Log error to console in development
    if (import.meta.env.DEV) {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // Call optional error callback
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const { children, fallback, title, description } = this.props;

    if (hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default error UI
      return (
        <Result
          status="error"
          title={title || 'Something went wrong'}
          subTitle={description || 'An unexpected error occurred. Please try again.'}
          extra={[
            <Button
              key="retry"
              type="primary"
              icon={<ReloadOutlined />}
              onClick={this.handleRetry}
            >
              Try Again
            </Button>,
            <Button key="reload" onClick={this.handleReload}>
              Reload Page
            </Button>,
          ]}
        >
          {import.meta.env.DEV && error && (
            <div style={{ textAlign: 'left', marginTop: 24 }}>
              <Collapse
                ghost
                items={[
                  {
                    key: 'error-details',
                    label: <Text type="secondary">Error Details (Development Only)</Text>,
                    children: (
                      <div>
                        <Paragraph>
                          <Text strong>Error: </Text>
                          <Text type="danger">{error.message}</Text>
                        </Paragraph>
                        {error.stack && (
                          <pre
                            style={{
                              background: '#f5f5f5',
                              padding: 12,
                              borderRadius: 4,
                              fontSize: 12,
                              overflow: 'auto',
                              maxHeight: 200,
                            }}
                          >
                            {error.stack}
                          </pre>
                        )}
                        {errorInfo?.componentStack && (
                          <>
                            <Paragraph style={{ marginTop: 16 }}>
                              <Text strong>Component Stack:</Text>
                            </Paragraph>
                            <pre
                              style={{
                                background: '#f5f5f5',
                                padding: 12,
                                borderRadius: 4,
                                fontSize: 12,
                                overflow: 'auto',
                                maxHeight: 200,
                              }}
                            >
                              {errorInfo.componentStack}
                            </pre>
                          </>
                        )}
                      </div>
                    ),
                  },
                ]}
              />
            </div>
          )}
        </Result>
      );
    }

    return children;
  }
}

export default ErrorBoundary;
