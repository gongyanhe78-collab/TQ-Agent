import js from "@eslint/js";
import pluginVue from "eslint-plugin-vue";
import globals from "globals";
import eslintConfigPrettier from "eslint-config-prettier";

export default [
  {
    ignores: ["dist", "node_modules"],
  },
  js.configs.recommended,
  ...pluginVue.configs["flat/essential"],
  {
    files: ["**/*.js", "**/*.vue"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
      },
    },
    rules: {
      // 未使用变量仅告警，允许 _ 前缀参数用于回调占位。
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      // 组件名不做强制多词限制，避免历史单文件组件被误报。
      "vue/multi-word-component-names": "off",
      // 聊天正文由 DOMPurify 清洗后再注入，允许 v-html 展示富文本回答。
      "vue/no-v-html": "off",
    },
  },
  eslintConfigPrettier,
];
