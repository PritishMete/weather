import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import dotenv from "dotenv";
import { genkit, z } from "genkit";
import { googleAI } from "@genkit-ai/google-genai";

dotenv.config();

// ─────────────────────────────────────────────
// Configuration - Using your provided keys
// ─────────────────────────────────────────────
const GOOGLE_API_KEY = "AIzaSyC1k58boI1RJ4XsUMaBI6cXErj2_3QPZPY";
const WEATHER_API_KEY = "fa5212911d9843c099a80832260204";
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// Initialize Genkit
// Use 'gemini-2.5-flash' for the best speed/stability in 2026
// ─────────────────────────────────────────────
const ai = genkit({
  plugins: [googleAI({ apiKey: GOOGLE_API_KEY })],
  model: "googleai/gemini-2.5-flash",
});

// ─────────────────────────────────────────────
// Zod Schemas
// ─────────────────────────────────────────────
const WeatherSchema = z.object({
  location: z.string(),
  country: z.string(),
  region: z.string(),
  localtime: z.string(),
  temp_c: z.number(),
  temp_f: z.number(),
  condition: z.string(),
  condition_code: z.number(),
  icon: z.string(),
  humidity: z.number(),
  wind_kph: z.number(),
  wind_dir: z.string(),
  feelslike_c: z.number(),
  feelslike_f: z.number(),
  uv: z.number(),
  visibility_km: z.number(),
  pressure_mb: z.number(),
  precip_mm: z.number(),
  cloud: z.number(),
  is_day: z.number(),
});

const FlowOutputSchema = z.object({
  weather: WeatherSchema,
  summary: z.string(),
  tips: z.array(z.string()),
});

// ─────────────────────────────────────────────
// Tool: Fetch Weather from WeatherAPI.com
// ─────────────────────────────────────────────
const getWeatherTool = ai.defineTool(
  {
    name: "getWeather",
    description: "Fetches real-time weather data for a location.",
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
      country: d.location.country,
      region: d.location.region,
      localtime: d.location.localtime,
      temp_c: d.current.temp_c,
      temp_f: d.current.temp_f,
      condition: d.current.condition.text,
      condition_code: d.current.condition.code,
      icon: "https:" + d.current.condition.icon,
      humidity: d.current.humidity,
      wind_kph: d.current.wind_kph,
      wind_dir: d.current.wind_dir,
      feelslike_c: d.current.feelslike_c,
      feelslike_f: d.current.feelslike_f,
      uv: d.current.uv,
      visibility_km: d.current.vis_km,
      pressure_mb: d.current.pressure_mb,
      precip_mm: d.current.precip_mm,
      cloud: d.current.cloud,
      is_day: d.current.is_day,
    };
  }
);

// ─────────────────────────────────────────────
// Genkit Flow: AI-powered weather analysis
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
      Analyze this weather for ${weather.location}: ${weather.temp_c}°C, ${weather.condition}.
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
        summary: `It is currently ${weather.condition} in ${weather.location}.`,
        tips: ["Check the local forecast.", "Dress for the weather.", "Have a nice day!"],
      };
    }

    return { weather, summary: parsed.summary, tips: parsed.tips };
  }
);

// ─────────────────────────────────────────────
// Express App
// ─────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json());

app.post("/weather", async (req: Request, res: Response) => {
  const { location } = req.body;
  if (!location) return res.status(400).json({ error: "Location is required" });

  try {
    const result = await weatherFlow({ location });
    res.json(result);
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🌤  AI Weather Backend running on http://localhost:${PORT}`);
});