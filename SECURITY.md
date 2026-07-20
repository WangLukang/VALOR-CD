# Security Policy

Do not report credentials, private dataset links, or access tokens in a public issue. Revoke any exposed token immediately.

Only load model checkpoints from trusted sources. PyTorch checkpoint files may contain serialized objects; verify their origin and checksum before loading them.
