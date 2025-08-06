#!/bin/bash
# GitHub推送脚本
# 请根据你的实际GitHub仓库信息修改以下变量

echo "🚀 Snake RL - GitHub推送脚本"
echo "=============================="

# 配置变量 (请修改为你的实际信息)
GITHUB_USERNAME="YOUR_USERNAME"
REPO_NAME="snake-rl-dqn"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "📋 推送配置:"
echo "   GitHub用户名: ${GITHUB_USERNAME}"
echo "   仓库名称: ${REPO_NAME}"
echo "   仓库URL: ${REPO_URL}"
echo ""

# 检查Git状态
echo "🔍 检查Git状态..."
git status

echo ""
echo "📊 检查提交历史..."
git log --oneline

echo ""
echo "🔗 检查远程仓库..."
git remote -v

# 如果没有远程仓库，添加它
if ! git remote | grep -q origin; then
    echo ""
    echo "📡 添加远程仓库..."
    echo "执行命令: git remote add origin ${REPO_URL}"
    echo "⚠️ 请手动执行此命令（替换为你的实际仓库URL）"
    echo ""
    echo "示例命令:"
    echo "git remote add origin https://github.com/yourusername/snake-rl-dqn.git"
else
    echo "✅ 远程仓库已配置"
fi

echo ""
echo "🚀 准备推送到GitHub..."
echo ""
echo "请按顺序执行以下命令:"
echo ""
echo "1️⃣ 添加远程仓库 (如果还没有):"
echo "git remote add origin https://github.com/YOUR_USERNAME/snake-rl-dqn.git"
echo ""
echo "2️⃣ 确保在main分支:"
echo "git branch -M main"
echo ""
echo "3️⃣ 推送到GitHub:"
echo "git push -u origin main"
echo ""
echo "📝 注意:"
echo "- 替换 YOUR_USERNAME 为你的GitHub用户名"
echo "- 替换 snake-rl-dqn 为你的实际仓库名"
echo "- 如果需要认证，GitHub会提示你输入密码或token"
echo ""
echo "🎉 推送成功后，你的项目将在GitHub上可见！"