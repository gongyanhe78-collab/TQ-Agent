import assert from "node:assert/strict";
import { buildDisplayedMessages } from "../frontend/src/messageVersions.js";

// 新版元数据应把两个回答折叠成一个槽，并默认显示最新版本。
const messages = [
  { role: "user", message_id: 1, content: "原问题" },
  { role: "assistant", message_id: 2, content: "版本一", version: 1, version_group_id: "user-1" },
  { role: "user", message_id: 3, content: "后续问题" },
  { role: "assistant", message_id: 4, content: "后续回答", version: 1, version_group_id: "user-3" },
  { role: "assistant", message_id: 5, content: "版本二", version: 2, version_group_id: "user-1", regenerated_from_message_id: 2 },
];

const latest = buildDisplayedMessages(messages, {});
assert.deepEqual(latest.map((item) => item.message_id), [1, 5, 3, 4]);
assert.deepEqual(latest[1].version_navigation, {
  current: 2,
  total: 2,
  can_previous: true,
  can_next: false,
});

// 用户选择第一版后，页面仍只有一个回答槽，但正文和操作目标都切回消息 2。
const first = buildDisplayedMessages(messages, { "user-1": 0 });
assert.deepEqual(first.map((item) => item.message_id), [1, 2, 3, 4]);
assert.equal(first[1].version_navigation.current, 1);
assert.equal(first[1].version_navigation.can_next, true);

// 旧数据缺少版本组时，仍应沿父消息编号把重生成回答归入同一组。
const legacy = [
  { role: "user", message_id: 10, content: "旧问题" },
  { role: "assistant", message_id: 11, content: "旧一版", version: 1 },
  { role: "assistant", message_id: 12, content: "旧二版", version: 2, regenerated_from_message_id: 11 },
];
const legacyDisplayed = buildDisplayedMessages(legacy, {});
assert.equal(legacyDisplayed.length, 2);
assert.equal(legacyDisplayed[1].message_id, 12);
assert.equal(legacyDisplayed[1].version_navigation.total, 2);

console.log("frontend message version grouping: OK");
