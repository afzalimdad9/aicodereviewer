import { GoogleGenAI } from '@google/genai';
import { z } from "zod";

let geminiClient: GoogleGenAI | null = null;

function getGeminiAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set");
  }

  if (!geminiClient) {
    geminiClient = new GoogleGenAI();
  }

  return geminiClient;
}

export const ReviewCommentSchema = z.object({
  file: z.string(),
  line: z.number(),
  severity: z.enum(["critical", "high", "medium", "low"]),
  category: z.enum(["bug", "security", "performance", "style", "suggestion"]),
  message: z.string(),
  suggestion: z.string().optional(),
});

export const ReviewResultSchema = z.object({
  summary: z.string(),
  riskScore: z.number().min(0).max(100),
  comments: z.array(ReviewCommentSchema),
});

export type ReviewComment = z.infer<typeof ReviewCommentSchema>;
export type ReviewResult = z.infer<typeof ReviewResultSchema>;

interface FileChange {
  filename: string;
  status: string;
  additions: number;
  deletions: number;
  patch?: string;
}

const SYSTEM_PROMPT = `You are an expert code reviewer. Analyze the provided pull request diff and provide a structured review.

Your review should:
1. Identify bugs, security issues, performance problems, and code style issues
2. Provide a brief summary of the changes
3. Assign a risk score (0-100) based on the complexity and potential issues
4. Give specific, actionable feedback with line numbers

Respond with valid JSON matching this schema:
{
  "summary": "Brief summary of changes and overall assessment",
  "riskScore": 0-100,
  "comments": [
    {
      "file": "path/to/file.ts",
      "line": 42,
      "severity": "critical" | "high" | "medium" | "low",
      "category": "bug" | "security" | "performance" | "style" | "suggestion",
      "message": "What the issue is",
      "suggestion": "How to fix it (optional)"
    }
  ]
}

Severity guide:
- critical: Security vulnerabilities, data loss, crashes
- high: Bugs that will cause issues in production
- medium: Should be fixed but won't break things
- low: Style issues, minor improvements

Be concise but specific. Reference exact line numbers from the diff.`;

export async function reviewCode(
  prTitle: string,
  files: FileChange[],
): Promise<ReviewResult> {
  const diffContent = files
    .filter((f) => f.patch)
    .map(
      (f) => `### ${f.filename} (${f.status})\n\`\`\`diff\n${f.patch}\n\`\`\``,
    )
    .join("\n\n");

  if (!diffContent.trim()) {
    return {
      summary: "No code changes to review (binary files or empty diff).",
      riskScore: 0,
      comments: [],
    };
  }

  const userPrompt = `Review this pull request:

**Title:** ${prTitle}

**Changes:**
${diffContent}`;

  const geminiAI = getGeminiAIClient();
  const response = await geminiAI.models.generateContent({
    model: "gemini-3.5-flash-lite",
    contents: `${SYSTEM_PROMPT}\n\n${userPrompt}`,
    config: {temperature: 0.3,maxOutputTokens: 2000},
  });

  const content = response.text;
  if (!content) {
    throw new Error("No response from AI");
  }

  const parsed = parseJson(content);
  const validated = ReviewResultSchema.parse(parsed);

  return validated;
}

function parseJson(input: string | null | object) {
  // Already a JS object/array
  if (typeof input === "object" && input !== null) {
    return input;
  }

  if (typeof input !== "string") {
    throw new TypeError("Input must be a JSON object, array, or string.");
  }

  const text = input.trim();

  // 1. Try parsing the entire input directly
  try {
    return JSON.parse(text);
  } catch {}

  // 2. Extract JSON from Markdown code fences
  const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);

  if (codeBlock) {
    try {
      return JSON.parse(codeBlock[1].trim());
    } catch {}
  }

  // 3. Find the first JSON object or array embedded in text
  const json = extractJson(text);

  if (json !== null) {
    try {
      return JSON.parse(json);
    } catch {}
  }

  throw new Error("No valid JSON object or array found.");
}

function extractJson(text: string) {
  let start = -1;
  let stack = [];
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    // Handle JSON strings
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    // Start of JSON object/array
    if (char === "{" || char === "[") {
      if (start === -1) {
        start = i;
      }

      stack.push(char);
      continue;
    }

    // End of JSON object/array
    if (char === "}" || char === "]") {
      if (stack.length === 0) continue;

      const expected = char === "}" ? "{" : "[";

      if (stack[stack.length - 1] !== expected) {
        return null;
      }

      stack.pop();

      // Complete JSON found
      if (stack.length === 0 && start !== -1) {
        return text.slice(start, i + 1);
      }
    }
  }

  return null;
}
