import { NextResponse } from 'next/server';

// Simplified middleware to just block the critical vulnerability
export function middleware(request) {
  // Only check for the dangerous x-middleware-subrequest header
  if (request.headers.get('x-middleware-subrequest')) {
    console.error('Blocked request with x-middleware-subrequest header');
    return new NextResponse(
      JSON.stringify({ success: false, message: 'Forbidden request' }),
      { status: 403, headers: { 'Content-Type': 'application/json' } }
    );
  }

  // Continue with the request if no issues found
  return NextResponse.next();
}

// Configure matcher only for API routes to minimize impact
export const config = {
  matcher: [
    '/api/:path*',
  ],
};