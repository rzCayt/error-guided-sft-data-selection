# GitHub 设置说明

## 仓库地址

当前远程仓库：

```text
https://github.com/rzCayt/error-guided-sft-data-selection
```

本地路径：

```powershell
E:\RA准备\07_error_guided_sft_repo
```

## 推送命令

```powershell
cd E:\RA准备\07_error_guided_sft_repo
git status --short --branch
git push
```

如果以后需要重新绑定远程仓库：

```powershell
cd E:\RA准备\07_error_guided_sft_repo
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/error-guided-sft-data-selection.git
git branch -M main
git push -u origin main
```

如果 SSH 已配置：

```powershell
cd E:\RA准备\07_error_guided_sft_repo
git remote add origin git@github.com:<YOUR_GITHUB_USERNAME>/error-guided-sft-data-selection.git
git branch -M main
git push -u origin main
```

## GitHub Pages 双语主页

仓库已新增：

```text
docs/index.html
```

这个页面提供中文/英文切换按钮。GitHub 的 README 不允许运行 JavaScript，因此真正的语言切换按钮放在 GitHub Pages 页面里。

启用方式：

1. 打开 GitHub 仓库页面。
2. 进入 `Settings`。
3. 左侧选择 `Pages`。
4. `Build and deployment` 选择 `Deploy from a branch`。
5. Branch 选择 `main`，folder 选择 `/docs`。
6. 保存后等待 GitHub Pages 构建完成。

启用后页面通常会出现在：

```text
https://rzcayt.github.io/error-guided-sft-data-selection/
```

如果仓库是 private，GitHub Pages 是否可见取决于账号和仓库设置。README 仍然会作为 GitHub 仓库首页默认展示。
