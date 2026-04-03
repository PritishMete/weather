import express, { Request, Response } from "express";
import cors from "cors";
import dotenv from "dotenv";
import { genkit, z } from "genkit";
import { googleAI } from "@genkit-ai/google-genai";

dotenv.config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const WEATHER_API_KEY = process.env.WEATHER_API_KEY;
const PORT = process.env.PORT || 3000;

const ai = genkit({
  plugins: [googleAI({ apiKey: GOOGLE_API_KEY })],
  model: "googleai/gemini-2.5-flash",
});

// --- Schemas ---
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

// --- Tool & Flow ---
const getWeatherTool = ai.defineTool(
  {
    name: "getWeather",
    description: "Fetches real-time weather data.",
    inputSchema: z.object({ location: z.string() }),
    outputSchema: WeatherSchema,
  },
  async ({ location }) => {
    const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${encodeURIComponent(location)}&aqi=no`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`WeatherAPI error: ${res.statusText}`);
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

const weatherFlow = ai.defineFlow(
  { name: "weatherFlow", inputSchema: z.object({ location: z.string() }) },
  async ({ location }) => {
    const weather = await getWeatherTool({ location });
    const { text } = await ai.generate({
      prompt: `Analyze weather for ${weather.location}: ${weather.temp_c}°C, ${weather.condition}. Respond ONLY JSON: {"summary": "2 sentences", "tips": ["tip1", "tip2", "tip3"]}`,
    });
    let parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
    return { weather, ...parsed };
  }
);

// --- Express App ---
const app = express();
app.use(cors());
app.use(express.json());

// 1. Weather Data Endpoint
app.post("/weather", async (req, res) => {
  try {
    const result = await weatherFlow({ location: req.body.location });
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 2. NEW: Location Suggestions Endpoint
app.get("/search", async (req, res) => {
  const query = req.query.q;
  if (!query) return res.json([]);
  try {
    const url = `https://api.weatherapi.com/v1/search.json?key=${WEATHER_API_KEY}&q=${query}`;
    const response = await fetch(url);
    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: "Search failed" });
  }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));