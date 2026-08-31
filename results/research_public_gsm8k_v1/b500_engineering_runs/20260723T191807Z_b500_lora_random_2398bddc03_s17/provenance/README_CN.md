# random-500 运行证据说明

本目录只保存能够从原始文件核验的证据，不补造中断进程没有写出的指标。

- `original_process.stdout.log` 和 `original_process.stderr.log` 是从工作区根目录按字节复制的原始日志，SHA-256 与源文件一致。
- `provenance_recovery.json` 记录原训练日志，以及 v1、v2、v3 三段可恢复评估之间的行数和哈希衔接。
- `tokenizer_warning_audit.json` 记录 tokenizer 警告的实际影响范围和冻结协议下的处理决定。

这些证据支持“random-500 的训练、adapter 保存、重新加载和 1,319 条评估工程闭环已经完成”。它们不支持三种 selector 已经比较，也不能替代原进程未保存的逐步训练遥测。
