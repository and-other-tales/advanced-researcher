export const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? (typeof window !== "undefined" ? window.location.origin : "http://localhost:8080");
