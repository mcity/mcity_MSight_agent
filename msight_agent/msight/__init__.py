"""MSight Vision live pipeline -- the agent's primary workflow.

runtime/   control plane, executors, graph diagnosis, node catalog (no MCP tools)
tools/     MCP tools the agent and dashboard call
handlers   chat-side tool handlers (consent, deferred recording)
hints      per-turn SESSION_STATE hint
routes     upload/download HTTP endpoints served by chat_server.py
prompts/   workflow prompt + reference docs
"""
