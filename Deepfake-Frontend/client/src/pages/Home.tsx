import { Link } from "wouter";
import { useEffect, useState } from "react";
import { ShieldAlert, Activity, FileVideo, FileAudio, Image as ImageIcon, ArrowRight, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, useStore, CrawlerStatus } from "@/lib/store";

const initialCrawler: CrawlerStatus = {
  enabled: false,
  running: false,
  last_run: null,
  last_error: null,
  records: 0,
  output: "",
};

export default function Home() {
  const { apiBaseUrl } = useStore();
  const [crawler, setCrawler] = useState<CrawlerStatus>(initialCrawler);

  useEffect(() => {
    let mounted = true;
    const pull = async () => {
      try {
        const status = await api.getCrawlerStatus(apiBaseUrl);
        if (mounted) setCrawler(status);
      } catch {
        // keep previous values
      }
    };
    pull();
    const t = setInterval(pull, 5000);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, [apiBaseUrl]);

  const stats = [
    { label: "Models Loaded", value: "3/3", status: "Active" },
    { label: "Crawler State", value: crawler.running ? (crawler.crawl_cycle_running ? "Running (Crawling)" : "Running (Standby)") : (crawler.enabled ? "Idle" : "Off"), status: crawler.last_error ? "Attention" : "Healthy" },
    { label: "Crawler Records", value: String(crawler.records || 0), status: crawler.last_run ? "Updated" : "Pending" },
  ];

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700 ease-out">
      <section className="relative overflow-hidden rounded-3xl border border-border bg-card p-8 md:p-12">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-transparent pointer-events-none" />
        <div className="relative z-10 max-w-2xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-mono">
            <ShieldAlert className="w-4 h-4" />
            <span>DeepGuard Core v2.1.0</span>
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-bold leading-tight">
            Advanced Multimodal <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-emerald-400">
              Deepfake Detection
            </span>
          </h1>
          <p className="text-lg text-muted-foreground leading-relaxed max-w-xl">
            Upload images, video, or audio files for instant deepfake analysis. Runtime crawler and feedback loops keep learning data fresh.
          </p>
          <div className="flex flex-wrap gap-4 pt-4">
            <Link href="/detect">
              <Button size="lg" className="font-mono gap-2" data-testid="button-start-detection">
                <Activity className="w-4 h-4" />
                Start Detection Run
                <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Link href="/settings">
              <Button size="lg" variant="outline" className="font-mono gap-2">
                <Bot className="w-4 h-4" />
                Open Crawler Controls
              </Button>
            </Link>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, i) => (
          <div key={i} className="p-6 rounded-2xl border border-border bg-card/50 backdrop-blur-sm space-y-2">
            <div className="text-sm font-mono text-muted-foreground uppercase tracking-wider">{stat.label}</div>
            <div className="text-3xl font-display font-semibold">{stat.value}</div>
            <div className="text-xs font-mono text-primary flex items-center gap-2 pt-2">
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              {stat.status}
            </div>
          </div>
        ))}
      </div>

      <section className="space-y-6">
        <h2 className="text-2xl font-display font-semibold border-b border-border pb-4">Supported Modalities</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { title: "Visual Analysis", desc: "Detects GAN-generated faces, spatial inconsistencies, and blending artifacts in still images.", icon: ImageIcon },
            { title: "Video Forensics", desc: "Frame-by-frame temporal analysis for lip-sync anomalies and subtle deepfake artifacts.", icon: FileVideo },
            { title: "Audio Authentication", desc: "Spectrogram analysis to identify AI-synthesized voices and cloned speech patterns.", icon: FileAudio },
          ].map((feature, i) => {
            const Icon = feature.icon;
            return (
              <div key={i} className="p-6 rounded-2xl border border-border bg-card hover:bg-accent/50 transition-colors group">
                <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                  <Icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{feature.desc}</p>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
