export const IST_TIME_ZONE = "Asia/Kolkata";

export function formatIst(isoText: string | null | undefined): string {
  if (!isoText) return "Never";
  const dt = new Date(isoText);
  if (Number.isNaN(dt.getTime())) return isoText;
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: IST_TIME_ZONE,
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(dt);
}
