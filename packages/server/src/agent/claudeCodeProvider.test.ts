import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessageEvent, Message, TranscriptContext } from "@earendil-works/pi-ai";
import {
	buildClaudeCodeRequest,
	ClaudeCodeTranslator,
	setClaudeCodeSessionCwdResolver,
} from "./claudeCodeProvider";

const model = { id: "sonnet", api: "claude-code-cli", provider: "claude-code" };
const user = (text: string): Message => ({ role: "user", content: text, timestamp: 0 });
const assistant = (text: string): Message =>
	({ role: "assistant", content: [{ type: "text", text }], timestamp: 0 }) as unknown as Message;
const context = (...messages: Message[]) => ({ messages }) as unknown as TranscriptContext;

afterEach(() => setClaudeCodeSessionCwdResolver(() => undefined));

function translate(lines: unknown[]) {
	const events: AssistantMessageEvent[] = [];
	let ended = false;
	const translator = new ClaudeCodeTranslator(
		{ push: (event) => events.push(event), end: () => (ended = true) },
		model,
	);
	for (const line of lines) translator.line(JSON.stringify(line));
	return { events, ended, output: translator.output };
}

test("a chat turn sends only the new user message and starts a Claude Code session named after the chat", () => {
	const cwd = mkdtempSync(join(tmpdir(), "cc-"));
	setClaudeCodeSessionCwdResolver(() => cwd);
	const id = "01a1838a-9aa6-718e-817e-b52b09cd9999";
	const request = buildClaudeCodeRequest(model, context(user("hi")), {
		sessionId: id,
		reasoning: "high",
	});
	expect(request.cwd).toBe(cwd);
	expect(request.args).toEqual(expect.arrayContaining(["--session-id", id, "--effort", "high"]));
	expect(request.args).not.toContain("--tools");
	expect(request.content).toEqual([{ type: "text", text: "hi" }]);
});

test("a chat that switched to Claude Code mid-way hands the earlier turns over once", () => {
	setClaudeCodeSessionCwdResolver(() => tmpdir());
	const request = buildClaudeCodeRequest(
		model,
		context(user("first"), assistant("answer"), user("second")),
		{ sessionId: "01a1838a-9aa6-718e-817e-b52b09cd8888" },
	);
	expect(request.content).toEqual([
		{ type: "text", text: "Earlier in this conversation:\n\nUser: first\n\nAssistant: answer" },
		{ type: "text", text: "second" },
	]);
});

test("compaction and sessionless calls run toolless and never touch the chat's Claude Code session", () => {
	setClaudeCodeSessionCwdResolver(() => tmpdir());
	const request = buildClaudeCodeRequest(model, context(user("summarize")), {
		sessionId: "01a1838a-9aa6-718e-817e-b52b09cd7777",
		cacheRetention: "none",
	});
	expect(request.args).toEqual(expect.arrayContaining(["--tools", "", "--no-session-persistence"]));
	expect(request.args).not.toContain("--session-id");
	expect(request.sessionId).toBeUndefined();
});

test("the CLI's stream becomes one assistant message with thinking, text and tool lines", () => {
	const fixture = readFileSync(join(import.meta.dir, "claudeCodeStream.fixture.jsonl"), "utf8")
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line) as unknown);
	const { events, ended, output } = translate(fixture);
	expect(ended).toBe(true);
	expect(events.at(-1)?.type).toBe("done");
	expect(output.content).toEqual([
		{ type: "thinking", thinking: "plan" },
		{ type: "text", text: "Running it.\n\n▸ **Bash** `echo hi`\n\n\n\nhi" },
	]);
	expect(output.usage.output).toBe(4);
	expect(output.usage.cost.total).toBe(0);
});

test("a failed CLI result ends the message as an error", () => {
	const { events, output } = translate([
		{ type: "result", subtype: "success", is_error: true, result: "usage limit reached" },
	]);
	expect(events.at(-1)?.type).toBe("error");
	expect(output.errorMessage).toBe("usage limit reached");
});
