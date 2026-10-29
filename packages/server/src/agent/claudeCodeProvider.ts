import { type ChildProcess, spawn } from "node:child_process";
import { existsSync, realpathSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { createInterface } from "node:readline";
import {
	type AssistantMessage,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	type ImageContent,
	type Message,
	type Model,
	type Provider,
	type SimpleStreamOptions,
	type StreamOptions,
	type TextContent,
	type ThinkingContent,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";

export const CLAUDE_CODE_PROVIDER = "claude-code";
const API = "claude-code-cli";
const STDERR_TAIL = 2000;

type Json = Record<string, unknown>;
type ClaudeBlock =
	| { type: "text"; text: string }
	| { type: "image"; source: { type: "base64"; media_type: string; data: string } };

let sessionCwd: (sessionId: string) => string | undefined = () => undefined;
export function setClaudeCodeSessionCwdResolver(
	resolver: (sessionId: string) => string | undefined,
): void {
	sessionCwd = resolver;
}

function model(id: string, name: string): Model<string> {
	return {
		id,
		name,
		api: API,
		provider: CLAUDE_CODE_PROVIDER,
		baseUrl: "",
		reasoning: true,
		thinkingLevelMap: {
			off: null,
			minimal: null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: "xhigh",
			max: "max",
		},
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	};
}

const MODELS = [
	model("opus", "Claude Opus"),
	model("sonnet", "Claude Sonnet"),
	model("haiku", "Claude Haiku"),
];

// shortcut: Windows finds only the native claude.exe, not an npm-installed claude.cmd shim.
function claudeOnPath(): boolean {
	const name = process.platform === "win32" ? "claude.exe" : "claude";
	return (process.env.PATH ?? "")
		.split(delimiter)
		.some((dir) => dir !== "" && existsSync(join(dir, name)));
}

const startedSessions = new Set<string>();

function claudeSessionExists(cwd: string, sessionId: string): boolean {
	if (startedSessions.has(sessionId)) return true;
	let real = cwd;
	try {
		real = realpathSync(cwd);
	} catch {}
	const configDir = process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude");
	return existsSync(
		join(configDir, "projects", real.replace(/[^a-zA-Z0-9]/g, "-"), `${sessionId}.jsonl`),
	);
}

function childEnv(): NodeJS.ProcessEnv {
	const env = { ...process.env };
	// An API key in the environment would make the CLI bill the API instead of the subscription.
	delete env.ANTHROPIC_API_KEY;
	delete env.ANTHROPIC_AUTH_TOKEN;
	return env;
}

function textOf(message: Message): string {
	if (typeof message.content === "string") return message.content;
	return message.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n");
}

function toClaudeBlocks(message: Message): ClaudeBlock[] {
	if (typeof message.content === "string") return [{ type: "text", text: message.content }];
	return message.content.flatMap((part): ClaudeBlock[] => {
		if (part.type === "text") return [{ type: "text", text: part.text }];
		if (part.type === "image") {
			const image = part as ImageContent;
			return [
				{ type: "image", source: { type: "base64", media_type: image.mimeType, data: image.data } },
			];
		}
		return [];
	});
}

function transcriptOf(messages: readonly Message[]): string {
	return messages
		.filter((message) => message.role === "user" || message.role === "assistant")
		.map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${textOf(message)}`)
		.join("\n\n");
}

export interface ClaudeCodeRequest {
	args: string[];
	cwd: string;
	content: ClaudeBlock[];
	sessionId?: string;
}

export function buildClaudeCodeRequest(
	model: Pick<Model<string>, "id">,
	context: Pick<TranscriptContext, "messages">,
	options: Pick<SimpleStreamOptions, "sessionId" | "reasoning" | "cacheRetention"> = {},
): ClaudeCodeRequest {
	const messages = context.messages.filter((message) => (message.role as string) !== "system");
	const args = [
		"-p",
		"--input-format",
		"stream-json",
		"--output-format",
		"stream-json",
		"--verbose",
		"--include-partial-messages",
		"--model",
		model.id,
		...(options.reasoning
			? ["--effort", options.reasoning === "minimal" ? "low" : options.reasoning]
			: []),
	];
	const cwd = options.sessionId ? sessionCwd(options.sessionId) : undefined;
	// pi's compaction and branch summaries reuse the chat's session id with caching off; they are
	// one-off completions and must not land in the Claude Code conversation.
	if (cwd === undefined || options.sessionId === undefined || options.cacheRetention === "none") {
		const system = context.messages
			.filter((message) => (message.role as string) === "system")
			.map((message) => textOf(message))
			.join("\n\n");
		return {
			args: [
				...args,
				"--tools",
				"",
				"--no-session-persistence",
				...(system ? ["--system-prompt", system] : []),
			],
			cwd: tmpdir(),
			content: [{ type: "text", text: transcriptOf(messages) }],
		};
	}
	const lastAssistant = messages.findLastIndex((message) => message.role === "assistant");
	const pending = messages.slice(lastAssistant + 1).filter((message) => message.role === "user");
	const resume = claudeSessionExists(cwd, options.sessionId);
	const earlier = resume ? "" : transcriptOf(messages.slice(0, lastAssistant + 1));
	return {
		args: [
			...args,
			"--permission-mode",
			"bypassPermissions",
			resume ? "--resume" : "--session-id",
			options.sessionId,
		],
		cwd,
		content: [
			...(earlier
				? [{ type: "text" as const, text: `Earlier in this conversation:\n\n${earlier}` }]
				: []),
			...(pending.length > 0
				? pending.flatMap(toClaudeBlocks)
				: [{ type: "text" as const, text: "Continue." }]),
		],
		sessionId: options.sessionId,
	};
}

function usageOf(raw: unknown): Usage {
	const usage = (raw ?? {}) as Json;
	const count = (key: string): number =>
		typeof usage[key] === "number" ? (usage[key] as number) : 0;
	const input = count("input_tokens");
	const output = count("output_tokens");
	const cacheRead = count("cache_read_input_tokens");
	const cacheWrite = count("cache_creation_input_tokens");
	return {
		input,
		output,
		cacheRead,
		cacheWrite,
		totalTokens: input + output + cacheRead + cacheWrite,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function toolLine(name: string, json: string): string {
	let input: Json = {};
	try {
		input = JSON.parse(json) as Json;
	} catch {}
	const detail = ["command", "file_path", "path", "pattern", "url", "query", "description"]
		.map((key) => input[key])
		.find((value): value is string => typeof value === "string");
	const summary = (detail ?? "").split("\n")[0]?.slice(0, 160).replaceAll("`", "'") ?? "";
	return `\n\n▸ **${name}**${summary ? ` \`${summary}\`` : ""}\n\n`;
}

export class ClaudeCodeTranslator {
	readonly output: AssistantMessage;
	private block: TextContent | ThinkingContent | undefined;
	private tool: { name: string; json: string } | undefined;
	finished = false;

	constructor(
		private readonly stream: Pick<AssistantMessageEventStream, "push" | "end">,
		model: Pick<Model<string>, "id" | "api" | "provider">,
	) {
		this.output = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: usageOf(undefined),
			stopReason: "stop",
			timestamp: Date.now(),
		};
		stream.push({ type: "start", partial: this.output });
	}

	line(raw: string): void {
		if (this.finished || raw.trim() === "") return;
		let event: Json;
		try {
			event = JSON.parse(raw) as Json;
		} catch {
			return;
		}
		if (event.parent_tool_use_id) return;
		if (event.type === "stream_event") this.streamEvent(event.event as Json);
		else if (event.type === "user") this.toolResults(event.message as Json);
		else if (event.type === "result") this.result(event);
	}

	fail(message: string, reason: "error" | "aborted" = "error"): void {
		if (this.finished) return;
		this.finished = true;
		this.close();
		this.output.stopReason = reason;
		this.output.errorMessage = message;
		this.stream.push({ type: "error", reason, error: this.output });
		this.stream.end();
	}

	private streamEvent(event: Json): void {
		const block = (event.content_block ?? {}) as Json;
		const delta = (event.delta ?? {}) as Json;
		switch (event.type) {
			case "message_start":
				this.output.usage = usageOf((event.message as Json | undefined)?.usage);
				break;
			case "message_delta":
				this.output.usage = usageOf(event.usage);
				break;
			case "content_block_start":
				if (block.type === "tool_use") this.tool = { name: String(block.name), json: "" };
				else if (block.type === "text" && this.block?.type === "text" && this.block.text)
					this.append("text", "\n\n");
				break;
			case "content_block_delta":
				if (delta.type === "text_delta") this.append("text", String(delta.text));
				else if (delta.type === "thinking_delta") this.append("thinking", String(delta.thinking));
				else if (delta.type === "input_json_delta" && this.tool)
					this.tool.json += String(delta.partial_json);
				break;
			case "content_block_stop":
				if (this.tool) this.append("text", toolLine(this.tool.name, this.tool.json));
				this.tool = undefined;
				break;
		}
	}

	private toolResults(message: Json): void {
		if (!Array.isArray(message.content)) return;
		for (const part of message.content as Json[]) {
			if (part.type !== "tool_result" || part.is_error !== true) continue;
			const text = typeof part.content === "string" ? part.content : JSON.stringify(part.content);
			this.append("text", `> ${text.split("\n")[0]?.slice(0, 200) ?? ""}\n\n`);
		}
	}

	private result(event: Json): void {
		if (event.is_error === true || event.subtype !== "success") {
			this.fail(
				typeof event.result === "string" && event.result ? event.result : String(event.subtype),
			);
			return;
		}
		this.finished = true;
		this.close();
		this.stream.push({ type: "done", reason: "stop", message: this.output });
		this.stream.end();
	}

	private append(kind: "text" | "thinking", delta: string): void {
		if (delta === "") return;
		if (this.block?.type !== kind) {
			this.close();
			this.block =
				kind === "text" ? { type: "text", text: "" } : { type: "thinking", thinking: "" };
			this.output.content.push(this.block);
			this.stream.push({
				type: `${kind}_start`,
				contentIndex: this.output.content.length - 1,
				partial: this.output,
			});
		}
		const contentIndex = this.output.content.length - 1;
		if (this.block.type === "text") {
			this.block.text += delta;
			this.stream.push({ type: "text_delta", contentIndex, delta, partial: this.output });
		} else {
			this.block.thinking += delta;
			this.stream.push({ type: "thinking_delta", contentIndex, delta, partial: this.output });
		}
	}

	private close(): void {
		const block = this.block;
		if (!block) return;
		this.block = undefined;
		const contentIndex = this.output.content.length - 1;
		if (block.type === "text")
			this.stream.push({
				type: "text_end",
				contentIndex,
				content: block.text,
				partial: this.output,
			});
		else
			this.stream.push({
				type: "thinking_end",
				contentIndex,
				content: block.thinking,
				partial: this.output,
			});
	}
}

function stop(child: ChildProcess): void {
	if (child.exitCode !== null || child.pid === undefined) return;
	try {
		if (process.platform === "win32") child.kill();
		else process.kill(-child.pid, "SIGTERM");
	} catch {}
}

function streamClaudeCode(
	model: Model<string>,
	context: TranscriptContext,
	options?: StreamOptions & Pick<SimpleStreamOptions, "reasoning">,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const translator = new ClaudeCodeTranslator(stream, model);
	const request = buildClaudeCodeRequest(model, context, options);
	const signal = options?.signal;
	if (signal?.aborted) {
		translator.fail("Request was aborted", "aborted");
		return stream;
	}
	const child = spawn("claude", request.args, {
		cwd: request.cwd,
		env: childEnv(),
		stdio: "pipe",
		detached: process.platform !== "win32",
	});
	if (request.sessionId) startedSessions.add(request.sessionId);
	let stderr = "";
	const onAbort = () => {
		translator.fail("Request was aborted", "aborted");
		stop(child);
	};
	signal?.addEventListener("abort", onAbort, { once: true });
	child.stderr.on("data", (chunk: Buffer) => {
		stderr = (stderr + chunk.toString()).slice(-STDERR_TAIL);
	});
	createInterface({ input: child.stdout }).on("line", (line) => translator.line(line));
	child.on("error", (error: NodeJS.ErrnoException) => {
		translator.fail(
			error.code === "ENOENT"
				? "The Claude Code CLI (`claude`) was not found on PATH."
				: error.message,
		);
	});
	child.on("close", (code) => {
		signal?.removeEventListener("abort", onAbort);
		translator.fail(stderr.trim() || `Claude Code exited with code ${code}`);
	});
	child.stdin.on("error", () => {});
	child.stdin.end(
		`${JSON.stringify({ type: "user", message: { role: "user", content: request.content } })}\n`,
	);
	return stream;
}

export const claudeCodeProvider: Provider = {
	id: CLAUDE_CODE_PROVIDER,
	name: "Claude Code",
	auth: {
		apiKey: {
			name: "Claude Code CLI",
			async resolve() {
				return claudeOnPath()
					? { auth: { apiKey: CLAUDE_CODE_PROVIDER }, source: "Claude Code CLI" }
					: undefined;
			},
		},
	},
	getModels: () => MODELS,
	stream: streamClaudeCode,
	streamSimple: streamClaudeCode,
};
