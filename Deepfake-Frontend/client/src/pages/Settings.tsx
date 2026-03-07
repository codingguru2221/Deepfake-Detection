import { useState, useEffect, useCallback } from "react";
import { Server, Settings2, Database, Bot, Play, Power, RefreshCw } from "lucide-react";
import { useStore, api, CrawlerStatus, RuntimeTrainingStatus } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

const initialCrawler: CrawlerStatus = {
  enabled: false,
  running: false,
  last_run: null,
  last_error: null,
  records: 0,
  output: "",
};

const initialTrain: RuntimeTrainingStatus = {
  running: false,
  last_run: null,
  last_error: null,
  last_result: null,
};

export default function Settings() {
  const { apiBaseUrl, setApiBaseUrl, settings, updateSettings } = useStore();
  const [urlInput, setUrlInput] = useState(apiBaseUrl);
  const [isChecking, setIsChecking] = useState(false);
  const [status, setStatus] = useState<"unknown" | "online" | "offline">("unknown");
  const [crawler, setCrawler] = useState<CrawlerStatus>(initialCrawler);
  const [crawlerLogs, setCrawlerLogs] = useState<string[]>([]);
  const [runtimeLogs, setRuntimeLogs] = useState<string[]>([]);
  const [training, setTraining] = useState<RuntimeTrainingStatus>(initialTrain);
  const [isCrawlerAction, setIsCrawlerAction] = useState(false);
  const [isTrainingAction, setIsTrainingAction] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    setUrlInput(apiBaseUrl);
  }, [apiBaseUrl]);

  const refreshCrawlerPanels = useCallback(async (targetBaseUrl?: string) => {
    const base = targetBaseUrl || apiBaseUrl;
    try {
      const [crawlerStatus, logData, trainingStatus] = await Promise.all([
        api.getCrawlerStatus(base),
        api.getCrawlerLogs(base, 100),
        api.getRuntimeTrainingStatus(base),
      ]);
      setCrawler(crawlerStatus);
      setCrawlerLogs(logData.lines || []);
      setRuntimeLogs(logData.runtimeLearning || []);
      setTraining(trainingStatus);
      setStatus("online");
    } catch {
      setStatus("offline");
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    refreshCrawlerPanels();
    const timer = setInterval(refreshCrawlerPanels, 5000);
    return () => clearInterval(timer);
  }, [refreshCrawlerPanels]);

  const handleSaveApi = async () => {
    setApiBaseUrl(urlInput);
    setIsChecking(true);
    setStatus("unknown");

    const isOnline = await api.checkHealth(urlInput);
    setStatus(isOnline ? "online" : "offline");
    setIsChecking(false);

    if (isOnline) {
      toast({
        title: "Connection Established",
        description: "Successfully connected to FastAPI backend.",
      });
      refreshCrawlerPanels(urlInput);
    } else {
      toast({
        title: "Connection Failed",
        description: "Could not reach backend. Using fallback mode.",
        variant: "destructive",
      });
    }
  };

  const handleCrawlerToggle = async (enabled: boolean) => {
    setIsCrawlerAction(true);
    try {
      await api.setCrawlerEnabled(apiBaseUrl, enabled);
      await refreshCrawlerPanels();
    } catch (err: any) {
      toast({ title: "Crawler update failed", description: err?.message || "Unknown error", variant: "destructive" });
    } finally {
      setIsCrawlerAction(false);
    }
  };

  const handleCrawlerRun = async () => {
    setIsCrawlerAction(true);
    try {
      const res = await api.runCrawler(apiBaseUrl);
      toast({ title: "Crawler", description: res.message });
      await refreshCrawlerPanels();
    } catch (err: any) {
      toast({ title: "Crawler run failed", description: err?.message || "Unknown error", variant: "destructive" });
    } finally {
      setIsCrawlerAction(false);
    }
  };

  const handleRuntimeTrain = async () => {
    setIsTrainingAction(true);
    try {
      const res = await api.runRuntimeTraining(apiBaseUrl, true);
      toast({
        title: "Runtime training",
        description: res.started ? "Training started in background." : (res.message || "Already running."),
      });
      await refreshCrawlerPanels();
    } catch (err: any) {
      toast({
        title: "Training failed",
        description: err?.message || "Unknown error",
        variant: "destructive",
      });
    } finally {
      setIsTrainingAction(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-5xl">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-display font-bold">Configuration</h1>
        <p className="text-muted-foreground">Manage API, crawler runtime, and model learning controls.</p>
      </div>

      <div className="grid gap-6">
        <div className="p-6 rounded-2xl border border-border bg-card space-y-6">
          <div className="flex items-center gap-3 border-b border-border pb-4">
            <Server className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-display font-medium">Backend Connection</h2>
          </div>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="apiUrl" className="font-mono text-xs uppercase tracking-wider text-muted-foreground">FastAPI Base URL</Label>
              <div className="flex gap-3">
                <Input
                  id="apiUrl"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  className="font-mono bg-background"
                  placeholder="http://localhost:8000"
                />
                <Button onClick={handleSaveApi} disabled={isChecking}>
                  {isChecking ? "Testing..." : "Apply & Test"}
                </Button>
              </div>
            </div>
            {status !== "unknown" && (
              <div className="p-4 rounded-lg bg-secondary/50 border border-border flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${status === "online" ? "bg-primary animate-pulse" : "bg-destructive"}`} />
                <span className="font-mono text-sm">
                  System Status: {status === "online" ? "CONNECTED" : "OFFLINE"}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="p-6 rounded-2xl border border-border bg-card space-y-6">
          <div className="flex items-center gap-3 border-b border-border pb-4">
            <Bot className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-display font-medium">Crawler Control</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border border-border bg-background/40 space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-base">Crawler Enabled</Label>
                <Switch checked={crawler.enabled} disabled={isCrawlerAction} onCheckedChange={handleCrawlerToggle} />
              </div>
              <div className="text-sm font-mono text-muted-foreground">
                State: {crawler.running ? "RUNNING" : "IDLE"} | Records: {crawler.records}
              </div>
              <div className="text-xs text-muted-foreground">Last run: {crawler.last_run || "Never"}</div>
              <div className="text-xs text-destructive">{crawler.last_error ? `Error: ${crawler.last_error}` : "No errors"}</div>
              <div className="flex gap-2 pt-2">
                <Button size="sm" onClick={handleCrawlerRun} disabled={isCrawlerAction || !crawler.enabled}>
                  <Play className="w-4 h-4 mr-2" />
                  Run Now
                </Button>
                <Button size="sm" variant="outline" onClick={() => refreshCrawlerPanels()}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-border bg-background/40 space-y-3">
              <Label className="text-base">Runtime Training</Label>
              <div className="text-sm font-mono text-muted-foreground">
                Status: {training.running ? "RUNNING" : "IDLE"}
              </div>
              <div className="text-xs text-muted-foreground">Last run: {training.last_run || "Never"}</div>
              <div className="text-xs text-destructive">{training.last_error ? `Error: ${training.last_error}` : "No errors"}</div>
              {training.last_result && (
                <div className="text-xs text-muted-foreground">
                  labeled={training.last_result.user_labeled_count || 0}, pseudo={training.last_result.pseudo_count || 0}, crawlerRefs={training.last_result.crawler_refs_count || 0}
                </div>
              )}
              <Button size="sm" onClick={handleRuntimeTrain} disabled={isTrainingAction || training.running}>
                <Power className="w-4 h-4 mr-2" />
                Train With Latest Data
              </Button>
            </div>
          </div>
        </div>

        <div className="p-6 rounded-2xl border border-border bg-card space-y-6">
          <div className="flex items-center gap-3 border-b border-border pb-4">
            <Settings2 className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-display font-medium">Pipeline Parameters</h2>
          </div>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-base">Multimodal Fusion</Label>
                <p className="text-sm text-muted-foreground">Automatically extract audio from video for combined analysis.</p>
              </div>
              <Switch checked={settings.fuseModalities} onCheckedChange={(c) => updateSettings({ fuseModalities: c })} />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-base">Auto-Run on Upload</Label>
                <p className="text-sm text-muted-foreground">Start inference immediately when a file is dropped.</p>
              </div>
              <Switch checked={settings.autoRun} onCheckedChange={(c) => updateSettings({ autoRun: c })} />
            </div>
          </div>
        </div>

        <div className="p-6 rounded-2xl border border-border bg-card space-y-6">
          <div className="flex items-center gap-3 border-b border-border pb-4">
            <Database className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-display font-medium">Crawler & Learning Logs</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Crawler Logs</Label>
              <div className="mt-2 h-56 overflow-auto rounded-lg border border-border bg-background p-3 text-xs font-mono space-y-1">
                {crawlerLogs.length === 0 ? <div className="text-muted-foreground">No crawler logs yet.</div> : crawlerLogs.map((ln, i) => <div key={i}>{ln}</div>)}
              </div>
            </div>
            <div>
              <Label className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Runtime Learning Logs</Label>
              <div className="mt-2 h-56 overflow-auto rounded-lg border border-border bg-background p-3 text-xs font-mono space-y-1">
                {runtimeLogs.length === 0 ? <div className="text-muted-foreground">No runtime-learning logs yet.</div> : runtimeLogs.map((ln, i) => <div key={i}>{ln}</div>)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
