import express, { Request, Response } from "express";
import cors from "cors";
import dotenv from "dotenv";
import { genkit, z } from "genkit";
import { googleAI } from "@genkit-ai/google-genai";

// Load variables from .env file
dotenv.config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const WEATHER_API_KEY = process.env.WEATHER_API_KEY;
const PORT = process.env.PORT || 3000;

// Validate keys are present
if (!GOOGLE_API_KEY || !WEATHER_API_KEY) {
  console.error("❌ Missing API Keys in .env file");
  process.exit(1);
}

// ─────────────────────────────────────────────
// Initialize Genkit
// ─────────────────────────────────────────────
const ai = genkit({
  plugins: [googleAI({ apiKey: GOOGLE_API_KEY })],
  model: "googleai/gemini-1.5-flash",
});

// ─────────────────────────────────────────────
// Zod Schemas
// ─────────────────────────────────────────────
const WeatherSchema = z.object({
  location: z.string(),
  temp_c: z.number(),
  condition: z.string(),
  humidity: z.number(),
  wind_kph: z.number(),
  icon: z.string(),
});

const FlowOutputSchema = z.object({
  weather: WeatherSchema,
  summary: z.string(),
  tips: z.array(z.string()),
});

// ─────────────────────────────────────────────
// Tool: Fetch Weather Data
// ─────────────────────────────────────────────
const getWeatherTool = ai.defineTool(
  {
    name: "getWeather",
    description: "Fetches real-time weather data",
    inputSchema: z.object({ location: z.string() }),
    outputSchema: WeatherSchema,
  },
  async ({ location }) => {
    const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${encodeURIComponent(location)}&aqi=no`;
    const res = await fetch(url);

    if (!res.ok) {
      throw new Error(`WeatherAPI error: ${res.statusText}`);
    }

    const d: any = await res.json();
    return {
      location: d.location.name,
      temp_c: d.current.temp_c,
      condition: d.current.condition.text,
      humidity: d.current.humidity,
      wind_kph: d.current.wind_kph,
      icon: "https:" + d.current.condition.icon,
    };
  }
);

// ─────────────────────────────────────────────
// Flow: AI Weather Analysis
// ─────────────────────────────────────────────
const weatherFlow = ai.defineFlow(
  {
    name: "weatherFlow",
    inputSchema: z.object({ location: z.string() }),
    outputSchema: FlowOutputSchema,
  },
  async ({ location }) => {
    const weather = await getWeatherTool({ location });

    const prompt = `
      You are a helpful weather assistant.
      Analyze: ${weather.location}, ${weather.temp_c}°C, ${weather.condition}.
      Respond ONLY with valid JSON:
      {
        "summary": "A friendly 2-sentence summary of the weather.",
        "tips": ["Tip 1", "Tip 2", "Tip 3"]
      }
    `;

    const { text } = await ai.generate({ prompt });

    let parsed;
    try {
      const clean = text.replace(/```json|```/g, "").trim();
      parsed = JSON.parse(clean);
    } catch {
      parsed = {
        summary: `It's currently ${weather.condition} in ${weather.location}.`,
        tips: ["Dress appropriately.", "Check for local updates.", "Have a great day!"],
      };
    }

    return { weather, ...parsed };
  }
);

// ─────────────────────────────────────────────
// Express Server Setup
// ─────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json());

// Health check endpoint for monitoring
app.get("/health", (req, res) => {
  res.json({ status: "online", timestamp: new Date().toISOString() });
});

// Main weather endpoint
app.post("/weather", async (req: Request, res: Response) => {
  const { location } = req.body;

  if (!location) {
    return res.status(400).json({ error: "Location is required" });
  }

  try {
    const result = await weatherFlow({ location });
    res.json(result);
  } catch (err: any) {
    console.error(`[Error]: ${err.message}`);
    res.status(500).json({ error: "Failed to process weather request" });
  }
});

// Listen on 0.0.0.0 to allow external connections
app.listen(Number(PORT), "0.0.0.0", () => {
  console.log(`\n🌤  Weather Backend Live`);
  console.log(`   Local:   http://localhost:${PORT}`);
  console.log(`   Network: http://10.220.207.63:${PORT}`);
});