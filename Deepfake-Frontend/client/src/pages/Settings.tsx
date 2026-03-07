import { useState, useEffect } from "react";
import { Server, Settings2, ShieldCheck, Database } from "lucide-react";
import { useStore, api } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

export default function Settings() {
  const { apiBaseUrl, setApiBaseUrl, settings, updateSettings } = useStore();
  const [urlInput, setUrlInput] = useState(apiBaseUrl);
  const [isChecking, setIsChecking] = useState(false);
  const [status, setStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const { toast } = useToast();

  useEffect(() => {
    setUrlInput(apiBaseUrl);
  }, [apiBaseUrl]);

  const handleSaveApi = async () => {
    setApiBaseUrl(urlInput);
    setIsChecking(true);
    setStatus('unknown');
    
    const isOnline = await api.checkHealth(urlInput);
    setStatus(isOnline ? 'online' : 'offline');
    setIsChecking(false);

    if (isOnline) {
      toast({
        title: "Connection Established",
        description: "Successfully connected to FastAPI backend.",
      });
    } else {
      toast({
        title: "Connection Failed",
        description: "Could not reach backend. Using mockup fallback mode.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-3xl">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-display font-bold">Configuration</h1>
        <p className="text-muted-foreground">Manage API connections and pipeline parameters.</p>
      </div>

      <div className="grid gap-6">
        
        {/* API Settings */}
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
              <p className="text-xs text-muted-foreground">
                Point this to your running Python FastAPI instance. If unreachable, the UI will use simulated mockup responses.
              </p>
            </div>

            {status !== 'unknown' && (
              <div className="p-4 rounded-lg bg-secondary/50 border border-border flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${status === 'online' ? 'bg-primary animate-pulse' : 'bg-destructive'}`} />
                <span className="font-mono text-sm">
                  System Status: {status === 'online' ? 'CONNECTED' : 'OFFLINE (MOCKUP MODE)'}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Inference Settings */}
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
              <Switch 
                checked={settings.fuseModalities}
                onCheckedChange={(c) => updateSettings({ fuseModalities: c })}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label className="text-base">Auto-Run on Upload</Label>
                <p className="text-sm text-muted-foreground">Start inference immediately when a file is dropped.</p>
              </div>
              <Switch 
                checked={settings.autoRun}
                onCheckedChange={(c) => updateSettings({ autoRun: c })}
              />
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="p-6 rounded-2xl border border-border bg-card space-y-6">
          <div className="flex items-center gap-3 border-b border-border pb-4">
            <Database className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-display font-medium">System Info</h2>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm font-mono">
            <div>
              <span className="text-muted-foreground block mb-1">UI Version</span>
              <span>v1.0.0-mockup</span>
            </div>
            <div>
              <span className="text-muted-foreground block mb-1">Environment</span>
              <span>{import.meta.env.MODE || 'development'}</span>
            </div>
            <div>
              <span className="text-muted-foreground block mb-1">Expected Endpoints</span>
              <ul className="text-xs text-muted-foreground mt-2 space-y-1">
                <li>POST /infer/image</li>
                <li>POST /infer/video</li>
                <li>POST /infer/audio</li>
                <li>POST /infer/multimodal</li>
                <li>GET /health</li>
              </ul>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
