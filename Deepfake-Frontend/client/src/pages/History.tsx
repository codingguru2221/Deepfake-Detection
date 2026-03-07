import { format } from "date-fns";
import { History as HistoryIcon, FileImage, FileVideo, FileAudio, Zap, Trash2 } from "lucide-react";
import { useStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function History() {
  const { history, clearHistory } = useStore();

  const getIcon = (modality: string) => {
    switch (modality) {
      case 'image': return FileImage;
      case 'video': return FileVideo;
      case 'audio': return FileAudio;
      default: return Zap;
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-display font-bold">Inference History</h1>
          <p className="text-muted-foreground">Log of all previously processed files and results.</p>
        </div>
        
        {history.length > 0 && (
          <Button variant="outline" size="sm" onClick={clearHistory} className="text-destructive hover:bg-destructive/10 hover:text-destructive">
            <Trash2 className="w-4 h-4 mr-2" />
            Clear Logs
          </Button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 border border-border border-dashed rounded-2xl bg-card/50 text-muted-foreground space-y-4">
          <HistoryIcon className="w-12 h-12 opacity-50" />
          <p className="font-mono text-sm">No analysis logs found in current session.</p>
        </div>
      ) : (
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs font-mono uppercase bg-secondary/50 text-muted-foreground border-b border-border">
                <tr>
                  <th className="px-6 py-4 font-medium">Timestamp</th>
                  <th className="px-6 py-4 font-medium">File</th>
                  <th className="px-6 py-4 font-medium">Modality</th>
                  <th className="px-6 py-4 font-medium">Prediction</th>
                  <th className="px-6 py-4 font-medium">Fake Prob</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {history.map((item) => {
                  const Icon = getIcon(item.modality);
                  const isFake = item.result.prediction === 'deepfake';
                  
                  return (
                    <tr key={item.id} className="hover:bg-accent/30 transition-colors">
                      <td className="px-6 py-4 font-mono text-xs whitespace-nowrap text-muted-foreground">
                        {format(new Date(item.timestamp), "MMM dd, HH:mm:ss")}
                      </td>
                      <td className="px-6 py-4 font-medium max-w-[200px] truncate" title={item.filename}>
                        {item.filename}
                      </td>
                      <td className="px-6 py-4 capitalize">
                        <div className="flex items-center gap-2">
                          <Icon className="w-4 h-4 text-muted-foreground" />
                          {item.modality}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={cn(
                          "px-2.5 py-1 rounded-full text-xs font-mono font-bold uppercase tracking-wider",
                          isFake 
                            ? "bg-destructive/10 text-destructive border border-destructive/20" 
                            : "bg-primary/10 text-primary border border-primary/20"
                        )}>
                          {item.result.prediction}
                        </span>
                      </td>
                      <td className="px-6 py-4 font-mono">
                        {(item.result.prob_fake * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
