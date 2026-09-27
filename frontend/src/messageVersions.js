// 为旧历史消息沿 regenerated_from_message_id 向上寻找根回答，兼容尚未保存版本组的会话。
export const resolveVersionGroup = (message, byId) => {
  if (message.version_group_id) return message.version_group_id;
  let current = message;
  const visited = new Set();
  while (current?.regenerated_from_message_id && !visited.has(current.regenerated_from_message_id)) {
    visited.add(current.regenerated_from_message_id);
    const parent = byId.get(current.regenerated_from_message_id);
    if (!parent) break;
    if (parent.version_group_id) return parent.version_group_id;
    current = parent;
  }
  return `answer-${current?.message_id || current?.local_id || message.local_id}`;
};

// 把同组回答折叠到第一次出现的位置，默认展示最新版本，并附加切换所需的导航状态。
export const buildDisplayedMessages = (messages, selectedVersionByGroup = {}) => {
  const byId = new Map(messages.filter((item) => item.message_id).map((item) => [item.message_id, item]));
  const groups = new Map();
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const groupId = resolveVersionGroup(message, byId);
    if (!groups.has(groupId)) groups.set(groupId, []);
    groups.get(groupId).push(message);
  }
  for (const versions of groups.values()) {
    versions.sort((left, right) => Number(left.version || 1) - Number(right.version || 1));
  }

  const emitted = new Set();
  const result = [];
  for (const message of messages) {
    if (message.role !== "assistant") {
      result.push(message);
      continue;
    }
    const groupId = resolveVersionGroup(message, byId);
    if (emitted.has(groupId)) continue;
    emitted.add(groupId);
    const versions = groups.get(groupId) || [message];
    const requestedIndex = selectedVersionByGroup[groupId];
    const selectedIndex = Number.isInteger(requestedIndex)
      ? Math.min(Math.max(requestedIndex, 0), versions.length - 1)
      : versions.length - 1;
    const selected = versions[selectedIndex];
    selected.version_group_id = groupId;
    selected.version_navigation = {
      current: selectedIndex + 1,
      total: versions.length,
      can_previous: selectedIndex > 0,
      can_next: selectedIndex < versions.length - 1,
    };
    result.push(selected);
  }
  return result;
};
