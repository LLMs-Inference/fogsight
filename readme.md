# Fogsight（雾象）[**中文**](./readme.md) | [**English**](./readme_en.md)


<p align="center">
  <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/logos/fogsight_logo_white_bg.png"
       alt="Fogsight Logo"
       width="300">
</p>

<h3 align="center">
雾象是一款由尖端大语言模型（Large Language Model，LLM）驱动的动画引擎 Agent
</h3>

<p align="center">
| <a href="https://deepwiki.com/fogsightai/fogsight"><b>🤖 Ask DeepWiki</b></a> | 
</p>

用户只需输入抽象概念或词语，雾象系统能自动将其转化为高水平的生动动画。部署项目代码到本地后，用户只需输入词语并点击生成，即可获得动画。

<p align="center">
  <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/1.png"
       alt="UI 截图"
       width="600">
</p>

我们设计了简洁易用的语言用户界面（Language User Interface，LUI），用户还可**进一步轻松编辑或改进生成的动画，实现言出法随的效果**。

雾象，意为 **“在模糊智能中的具象”**。*它是 WaytoAGI 开源计划项目的成员。 WaytoAGI 致力于让更多人因 AI 而强大*。


## 💻 动画示例

以下是 Fogsight AI 生成的动画示例，请点击跳转查看：


<table>
  <tr>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1PXgKzBEyN">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/entropy_increase_thumbnail.png" width="350"><br>
        <strong>The Law of Increasing Entropy (Physics)</strong><br>
        <strong>熵增定律（物理学）</strong><br>
        <em>输入：熵增定律</em>
      </a>
    </td>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1yXgKzqE42">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/euler_formula_thumbnail.png" width="350"><br>
        <strong>Euler's Polyhedron Formula (Mathematics)</strong><br>
        <strong>欧拉多面体定理（数学）</strong><br>
        <em>输入：欧拉定理</em>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1sQgKzMEox">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/bubble_sort_thumbnail.png" width="350"><br>
        <strong>Bubble Sort (Computer Science)</strong><br>
        <strong>冒泡排序（计算机科学）</strong><br>
        <em>输入：冒泡排序</em>
      </a>
    </td>
    <td align="center">
      <a href="https://www.bilibili.com/video/BV1yQgKzMEo6">
        <img src="https://github.com/hamutama/caimaopics/raw/main/fogsight/thumbnails/affordance_thumbnail.png" width="350"><br>
        <strong>Affordance (Design)</strong><br>
        <strong>可供性（设计学）</strong><br>
        <em>输入：affordance in design</em>
      </a>
    </td>
  </tr>
</table>


## 🛞 核心功能

* **概念即影像**：输入一个具体主题，Fogsight 将为您精心打造一部叙事完整、高水平的动画。这部动画将配备双语旁白，并呈现电影级的视觉质感。

* **智能编排**：Fogsight 的核心在于其强大的 LLM 驱动的编排能力。从旁白到视觉元素，再到动态效果，AI 能够自动完成整个创作流程，流畅无阻。

* **语言用户界面（LUI）**：通过多轮与 AI 的对话，您可以对动画进行精确调整和优化，直至实现心中最理想的艺术效果。  


## 🙌 快速上手

### 环境要求

* Python 3.9+

* 一个现代网络浏览器，如 Chrome、Firefox 或 Edge。

* 拥有调用权限的大语言模型的 API KEY。我们推荐您使用 Google Gemini Pro 2.5。 

### 安装与运行

1. **克隆代码仓库：**
   ```bash
   git clone https://github.com/fogsightai/fogsight.git
   cd fogsight
   ```

2. **安装依赖库：**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置 API KEY 和 BASE URL：**

   ```bash
   cp demo-credentials.json credentials.json
   # 复制 demo-credentials.json 文件并重命名为 credentials.json
   # 编辑 credentials.json 文件，填入您的 API_KEY 和 BASE_URL。
   # **请注意**，fogsight 实际使用的是与 OpenAI 兼容的软件开发工具包（Software Development Kit，SDK），但您仍应使用 Gemini 2.5 Pro。
   ```

4. **一键启动应用程序：**

   ```bash
   python start_fogsight.py
   # 运行 start_fogsight.py 脚本
   # 它将自动启动后端服务并在浏览器中自动打开 http://127.0.0.1:8000
   ```

5. **开始创作！**

   在页面中输入一个主题（例如 “冒泡排序”），然后等待结果生成即可。


## 📱 联系我们/加入群聊

请访问 [此链接](https://fogsightai.feishu.cn/wiki/WvODwyUr1iSAe0kEyKfcpqvynGc?from=from_copylink) 联系我们或加入交流群。


## 👨‍💻 贡献者

### 高校

* [@taited](https://taited.github.io/) - 香港中文大学（深圳） 博士生

* [@yjydya](https://github.com/ydyjya) - 南洋理工大学 博士生

### WaytoAGI 社区

* [@richkatchen 陈财猫](https://okjk.co/enodyA) - WaytoAGI 社区成员

* [@kk](https://okjk.co/zC8myE) - WaytoAGI 社区成员

### Index Future Lab

* [何淋 (@Lin he)](https://github.com/zerohe2001)

### AI 探索家

* [黄小刀 (@Xiaodao Huang)](https://okjk.co/CkFav6)

### 独立开发者与 AI 艺术家

* [@shuyan-5200](https://github.com/shuyan-5200)

* [王如玥 (@Ruyue Wang)](https://github.com/Moonywang)

* [@Jack-the-Builder](https://github.com/Jack-the-Builder)

* [@xiayurain95](https://github.com/xiayurain95)

* [@Lixin Cai 蔡李鑫](https://github.com/Lixin-Cai)


## 💓 开源许可

本项目实际基于 MIT 许可证开源。如能在引用此项目时署上我们的名字并提供项目链接，我们将不胜感激 ✨😊。
