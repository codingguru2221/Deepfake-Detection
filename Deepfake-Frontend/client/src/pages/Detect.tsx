import { useState, useRef } from "react";
import { Upload, FileImage, FileVideo, FileAudio, X, AlertCircle, CheckCircle2, ShieldAlert, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Modality, useStore, api } from "@/lib/store";
import { cn } from "@/lib/utils";

export default function Detect() {
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState<Modality>('image');
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { apiBaseUrl, addHistory, settings } = useStore();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFileSelection(e.target.files[0]);
    }
  };

  const processFileSelection = (selectedFile: File) => {
    setError(null);
    setResult(null);
    
    // Auto-detect modality based on file type
    const type = selectedFile.type;
    let detectedModality: Modality = modality;
    
    if (type.startsWith('image/')) detectedModality = 'image';
    else if (type.startsWith('video/')) detectedModality = 'video';
    else if (type.startsWith('audio/')) detectedModality = 'audio';
    else {
      setError("Unsupported file format. Please upload an image, video, or audio file.");
      return;
    }

    setFile(selectedFile);
    
    // Auto-switch to multimodal if settings say so, otherwise use detected
    if (settings.fuseModalities && detectedModality === 'video') {
      setModality('multimodal'); // Videos often have audio
    } else {
      setModality(detectedModality);
    }
  };

  const runInference = async () => {
    if (!file) return;

    setIsProcessing(true);
    setProgress(0);
    setError(null);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 15, 90));
    }, 200);

    try {
      const prediction = await api.infer(apiBaseUrl, modality, file);
      
      clearInterval(interval);
      setProgress(100);
      setResult(prediction);
      
      addHistory({
        id: Math.random().toString(36).substring(7),
        filename: file.name,
        timestamp: new Date().toISOString(),
        modality,
        result: prediction
      });
      
    } catch (err: any) {
      clearInterval(interval);
      setProgress(0);
      setError(err.message || "Failed to process the file.");
    } finally {
      setTimeout(() => setIsProcessing(false), 500);
    }
  };

  const modalities: { id: Modality, label: string, icon: any }[] = [
    { id: 'image', label: 'Image', icon: FileImage },
    { id: 'video', label: 'Video', icon: FileVideo },
    { id: 'audio', label: 'Audio', icon: FileAudio },
    { id: 'multimodal', label: 'Multimodal', icon: Zap },
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-display font-bold">Detection Pipeline</h1>
        <p className="text-muted-foreground">Upload media for deepfake analysis through the neural engine.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Upload & Controls */}
        <div className="space-y-6">
          <div className="p-1 bg-secondary rounded-lg inline-flex">
            {modalities.map((m) => {
              const Icon = m.icon;
              const isActive = modality === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setModality(m.id)}
                  disabled={isProcessing}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all",
                    isActive 
                      ? "bg-card text-foreground shadow-sm" 
                      : "text-muted-foreground hover:text-foreground",
                    isProcessing && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {m.label}
                </button>
              );
            })}
          </div>

          <div
            className={cn(
              "relative overflow-hidden border-2 border-dashed rounded-2xl p-12 transition-all duration-200 flex flex-col items-center justify-center text-center group cursor-pointer",
              isDragging ? "border-primary bg-primary/5" : "border-border bg-card hover:border-primary/50 hover:bg-accent/30",
              file && "border-primary/50 bg-accent/20"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept="image/*,video/*,audio/*"
            />
            
            {file ? (
              <div className="w-full relative z-10 space-y-4">
                <div className="w-16 h-16 mx-auto rounded-xl bg-secondary flex items-center justify-center">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <p className="font-mono text-sm truncate px-4">{file.name}</p>
                  <p className="text-xs text-muted-foreground mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
                <div className="flex justify-center pt-2">
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null); }}
                    disabled={isProcessing}
                  >
                    <X className="w-4 h-4 mr-2" /> Clear Selection
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4 pointer-events-none">
                <div className="w-16 h-16 mx-auto rounded-full bg-secondary flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
                <div>
                  <p className="font-medium">Drag & drop your file here</p>
                  <p className="text-sm text-muted-foreground mt-1">or click to browse from your computer</p>
                </div>
                <div className="text-xs font-mono text-muted-foreground/60 pt-4">
                  Supports .JPG, .PNG, .MP4, .WAV, .MP3
                </div>
              </div>
            )}
            
            {isProcessing && (
              <div className="absolute inset-0 bg-background/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center p-8">
                <div className="w-full max-w-xs space-y-4">
                  <div className="flex justify-between text-sm font-mono">
                    <span className="text-primary animate-pulse">Running Neural Inference...</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-1" />
                </div>
              </div>
            )}
          </div>

          <Button 
            className="w-full h-12 text-lg font-mono font-bold tracking-tight"
            disabled={!file || isProcessing}
            onClick={runInference}
            data-testid="button-run-inference"
          >
            {isProcessing ? "Processing..." : "INITIALIZE SCAN"}
          </Button>

          {error && (
            <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-start gap-3">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* Right Column: Results */}
        <div className="bg-card border border-border rounded-2xl overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-border bg-secondary/30">
            <h3 className="font-display font-medium">Analysis Results</h3>
          </div>
          
          <div className="flex-1 p-6 flex flex-col">
            {!result ? (
              <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground space-y-4 opacity-50">
                <ShieldAlert className="w-12 h-12" />
                <p className="text-sm font-mono text-center">Awaiting data input.<br/>System idle.</p>
              </div>
            ) : (
              <div className="space-y-8 animate-in slide-in-from-bottom-4 fade-in duration-500">
                
                {/* Main Prediction Badge */}
                <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-border bg-secondary/20 relative overflow-hidden">
                  {result.prediction === 'deepfake' ? (
                    <div className="absolute inset-0 bg-destructive/5" />
                  ) : (
                    <div className="absolute inset-0 bg-primary/5" />
                  )}
                  
                  <div className="relative z-10 text-center space-y-2">
                    <div className="text-sm font-mono text-muted-foreground uppercase tracking-widest">Final Verdict</div>
                    <div className={cn(
                      "text-5xl font-display font-bold tracking-tight uppercase",
                      result.prediction === 'deepfake' ? "text-destructive" : "text-primary"
                    )}>
                      {result.prediction}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg border border-border bg-card">
                    <div className="text-xs font-mono text-muted-foreground mb-1">Fake Probability</div>
                    <div className="text-2xl font-mono">{(result.prob_fake * 100).toFixed(1)}%</div>
                  </div>
                  <div className="p-4 rounded-lg border border-border bg-card">
                    <div className="text-xs font-mono text-muted-foreground mb-1">Confidence Score</div>
                    <div className="text-2xl font-mono">{(result.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>

                {/* Multimodal Details */}
                {result.modalityScores && (
                  <div className="space-y-4">
                    <h4 className="text-sm font-mono text-muted-foreground border-b border-border pb-2">Per-Modality Breakdown</h4>
                    <div className="space-y-3">
                      {Object.entries(result.modalityScores).map(([mod, score]) => (
                        <div key={mod} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="capitalize">{mod}</span>
                            <span className="font-mono">{(Number(score) * 100).toFixed(1)}% Fake</span>
                          </div>
                          <Progress value={Number(score) * 100} className="h-1.5" />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {result.details?.processing_time && (
                  <div className="pt-4 mt-auto border-t border-border flex justify-between items-center text-xs font-mono text-muted-foreground">
                    <span>Model: {result.details.model_version}</span>
                    <span>Time: {result.details.processing_time}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
