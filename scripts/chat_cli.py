"""
Interactive chat CLI for nanochat_mlx.
Ported from nanochat/scripts/chat_cli.py (PyTorch) to Apple MLX.

Key differences from PyTorch version:
- No torch.amp.autocast
- No device management (MLX unified memory)
- Uses Engine for generation (handles KV cache, sampling)
- Supports --prompt for single response or interactive mode

Usage:
    # Interactive mode
    python scripts/chat_cli.py --checkpoint runs/chat_sft_xxx/final

    # Single prompt mode
    python scripts/chat_cli.py --checkpoint runs/chat_sft_xxx/final --prompt "What is 2+2?"

    # With generation parameters
    python scripts/chat_cli.py --checkpoint runs/chat_sft_xxx/final --temperature 0.7 --max-tokens 512
"""

import os
import sys
import argparse
import time

import mlx.core as mx

from nanochat_mlx.common import mlx_init, print0
from nanochat_mlx.engine import Engine
from nanochat_mlx.checkpoint_manager import load_model
from nanochat_mlx.tokenizer import get_tokenizer


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def color(text, c):
    return f"{c}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages a multi-turn chat conversation."""

    def __init__(self, engine, tokenizer, max_tokens=1024, temperature=0.7,
                 top_k=50, system_prompt=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.messages = []

        if system_prompt:
            # System messages get folded into the first user message by the tokenizer
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content):
        """Add a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})

    def generate_response(self, stream=True):
        """
        Generate an assistant response for the current conversation.

        Args:
            stream: If True, yield tokens as they are generated

        Returns:
            The complete assistant response string
        """
        # Build the conversation so far (without the assistant response)
        conversation = {"messages": self.messages.copy()}

        # Add empty assistant message to prime for completion
        conversation["messages"].append({"role": "assistant", "content": ""})

        # Tokenize the conversation for completion
        prompt_ids = self.tokenizer.render_for_completion(conversation)

        # Get special token IDs for stop conditions
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        user_start = self.tokenizer.encode_special("<|user_start|>")
        stop_tokens = [assistant_end]
        if user_start is not None:
            stop_tokens.append(user_start)

        # Generate using Engine's streaming API
        # Engine.generate yields (token_column, token_masks) for num_samples=1
        response_tokens = []
        response_text = ""

        gen = self.engine.generate(
            prompt_ids,
            num_samples=1,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        for token_column, token_masks in gen:
            token_id = token_column[0]
            # Stop on special end tokens
            if token_id in stop_tokens:
                break
            response_tokens.append(token_id)

            if stream:
                # Decode incrementally
                new_text = self.tokenizer.decode(response_tokens)
                delta = new_text[len(response_text):]
                if delta:
                    print(delta, end="", flush=True)
                response_text = new_text

        if stream:
            print()  # newline after streaming
        else:
            response_text = self.tokenizer.decode(response_tokens)
            print(response_text)

        # Add the assistant response to the conversation history
        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    def reset(self):
        """Clear conversation history."""
        # Preserve system prompt if any
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(session):
    """Run an interactive chat session."""
    print()
    print(color("=" * 60, Colors.CYAN))
    print(color("  NanoChat MLX - Interactive Chat", Colors.BOLD + Colors.CYAN))
    print(color("=" * 60, Colors.CYAN))
    print()
    print(color("Commands:", Colors.YELLOW))
    print(color("  /quit, /exit  - Exit the chat", Colors.DIM))
    print(color("  /reset, /new  - Start a new conversation", Colors.DIM))
    print(color("  /system <msg> - Set a system prompt and reset", Colors.DIM))
    print(color("  /temp <val>   - Set temperature (current: {:.1f})".format(
        session.temperature), Colors.DIM))
    print(color("  /tokens <val> - Set max tokens (current: {})".format(
        session.max_tokens), Colors.DIM))
    print()

    while True:
        try:
            user_input = input(color("You: ", Colors.GREEN + Colors.BOLD))
        except (EOFError, KeyboardInterrupt):
            print()
            print(color("Goodbye!", Colors.CYAN))
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            cmd_args = user_input[len(cmd):].strip()

            if cmd in ("/quit", "/exit", "/q"):
                print(color("Goodbye!", Colors.CYAN))
                break
            elif cmd in ("/reset", "/new", "/clear"):
                session.reset()
                print(color("Conversation reset.", Colors.YELLOW))
                continue
            elif cmd == "/system":
                if cmd_args:
                    session.reset()
                    session.messages.insert(0, {"role": "system", "content": cmd_args})
                    print(color(f"System prompt set: {cmd_args}", Colors.YELLOW))
                else:
                    print(color("Usage: /system <message>", Colors.RED))
                continue
            elif cmd == "/temp":
                try:
                    session.temperature = float(cmd_args)
                    print(color(f"Temperature set to {session.temperature}", Colors.YELLOW))
                except ValueError:
                    print(color("Usage: /temp <float>", Colors.RED))
                continue
            elif cmd == "/tokens":
                try:
                    session.max_tokens = int(cmd_args)
                    print(color(f"Max tokens set to {session.max_tokens}", Colors.YELLOW))
                except ValueError:
                    print(color("Usage: /tokens <int>", Colors.RED))
                continue
            else:
                print(color(f"Unknown command: {cmd}", Colors.RED))
                continue

        # Add user message and generate response
        session.add_user_message(user_input)

        print(color("Assistant: ", Colors.BLUE + Colors.BOLD), end="", flush=True)
        t0 = time.time()
        response = session.generate_response(stream=True)
        dt = time.time() - t0

        # Show timing info
        n_tokens = len(session.tokenizer.encode(response))
        tps = n_tokens / dt if dt > 0 else 0
        print(color(f"  [{n_tokens} tokens, {dt:.1f}s, {tps:.1f} tok/s]",
                     Colors.GRAY))
        print()


# ---------------------------------------------------------------------------
# Single prompt mode
# ---------------------------------------------------------------------------

def run_single_prompt(session, prompt):
    """Run a single prompt and print the response."""
    session.add_user_message(prompt)

    t0 = time.time()
    response = session.generate_response(stream=True)
    dt = time.time() - t0

    n_tokens = len(session.tokenizer.encode(response))
    tps = n_tokens / dt if dt > 0 else 0
    print(color(f"\n[{n_tokens} tokens, {dt:.1f}s, {tps:.1f} tok/s]",
                Colors.GRAY), file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoChat MLX Chat CLI")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output")

    args = parser.parse_args()

    # Initialize
    mlx_init(seed=args.seed)

    # Load model
    print0(f"Loading model from: {args.checkpoint}")
    t0 = time.time()
    model, config = load_model(args.checkpoint)
    load_time = time.time() - t0
    print0(f"Model loaded in {load_time:.1f}s")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Create engine for generation
    engine = Engine(model=model, tokenizer=tokenizer)

    # Create chat session
    session = ChatSession(
        engine=engine,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        system_prompt=args.system,
    )

    # Run
    if args.prompt:
        run_single_prompt(session, args.prompt)
    else:
        run_interactive(session)


if __name__ == "__main__":
    main()
