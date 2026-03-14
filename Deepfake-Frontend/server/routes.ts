import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

function getBackendBaseUrl(): string {
  return (process.env.FASTAPI_BASE_URL || process.env.VITE_API_BASE_URL || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
}

function getForwardHeaders(req: Request): Headers {
  const headers = new Headers();

  for (const [key, value] of Object.entries(req.headers)) {
    if (!value) continue;
    if (["host", "connection", "content-length"].includes(key.toLowerCase())) continue;

    if (Array.isArray(value)) {
      for (const item of value) headers.append(key, item);
    } else {
      headers.set(key, value);
    }
  }

  return headers;
}

function getRequestBody(req: Request): { body?: BodyInit; isStream: boolean } {
  if (req.method === "GET" || req.method === "HEAD") return { body: undefined, isStream: false };

  const contentType = String(req.headers["content-type"] || "").toLowerCase();

  if (contentType.includes("multipart/form-data")) {
    return { body: req as unknown as BodyInit, isStream: true };
  }

  if (req.rawBody && Buffer.isBuffer(req.rawBody)) {
    return { body: req.rawBody, isStream: false };
  }

  if (contentType.includes("application/json")) {
    return { body: JSON.stringify(req.body ?? {}), isStream: false };
  }

  if (contentType.includes("application/x-www-form-urlencoded")) {
    return { body: new URLSearchParams(req.body ?? {}).toString(), isStream: false };
  }

  return { body: undefined, isStream: false };
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // put application routes here
  // prefix all routes with /api

  // use storage to perform CRUD operations on the storage interface
  // e.g. storage.insertUser(user) or storage.getUserByUsername(username)

  app.all("/api/*path", async (req, res, next) => {
    const backendUrl = new URL(`/${req.params.path.join("/")}${req.url.includes("?") ? req.url.slice(req.url.indexOf("?")) : ""}`, `${getBackendBaseUrl()}/`);
    const headers = getForwardHeaders(req);
    const { body, isStream } = getRequestBody(req);

    try {
      const requestInit: RequestInit & { duplex?: "half" } = {
        method: req.method,
        headers,
        body,
      };
      if (isStream) {
        requestInit.duplex = "half";
      }

      const response = await fetch(backendUrl, requestInit);

      res.status(response.status);
      response.headers.forEach((value, key) => {
        if (["content-encoding", "transfer-encoding", "connection"].includes(key.toLowerCase())) return;
        res.setHeader(key, value);
      });

      const arrayBuffer = await response.arrayBuffer();
      res.send(Buffer.from(arrayBuffer));
    } catch (error) {
      next(error);
    }
  });

  return httpServer;
}
