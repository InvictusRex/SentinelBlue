export function telemetryEndpointFromInput(value: string, pageProtocol = window.location.protocol): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "";

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  const url = new URL(withProtocol);
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("unsupported protocol");

  const loopback = url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "[::1]";
  if (pageProtocol === "https:" && url.protocol === "http:" && !loopback) {
    throw new Error("Use an HTTPS telemetry URL from this hosted dashboard.");
  }
  if (url.pathname === "/" || url.pathname === "") url.pathname = "/telemetry.json";
  url.search = "";
  url.hash = "";
  return url.toString();
}
