<template>
  <section class="smart-result">
    <div v-if="queryTitle || queryValues.length || queryDetail" class="query-overview">
      <strong v-if="queryTitle">{{ queryTitle }}</strong>
      <div v-if="queryValues.length" class="query-chips">
        <span v-for="value in queryValues" :key="value">{{ value }}</span>
      </div>
      <p v-if="queryDetail" class="query-detail">{{ queryDetail }}</p>
    </div>

    <div class="case-list">
      <article v-for="item in result.matched_cases || []" :key="item.case_id" class="case-card">
        <header><h3>{{ item.rank }}. {{ item.title }}</h3><strong>{{ percent(item.retrieval_score) }}</strong></header>
        <p class="meta">{{ [item.date_range, item.source_pdf].filter(Boolean).join(" · ") }}</p>
        <div class="tags"><span v-for="tag in [...(item.disaster_types || []), ...(item.affected_areas || [])]" :key="tag">{{ tag }}</span></div>
        <div class="scores">
          <span v-for="score in scoreItems(item)" :key="score.label">{{ score.label }} {{ percent(score.value) }}</span>
        </div>
        <h4>匹配理由</h4><InfoList :values="item.match_reasons" />
        <h4>参考经验</h4>
        <div v-if="item.key_references?.length" class="reference-list">
          <article v-for="(reference, referenceIndex) in item.key_references" :key="`${item.case_id}-reference-${referenceIndex}`" class="reference-item">
            <p class="reference-text"><span class="reference-marker">{{ referenceIndex + 1 }}</span>{{ referenceText(reference) }}</p>
            <div class="evidence-actions">
              <span v-if="referenceChunkIds(reference).length" class="evidence-label">文字证据</span>
              <button
                v-for="chunkId in referenceChunkIds(reference)"
                :key="`${item.case_id}-${chunkId}`"
                class="evidence-button"
                type="button"
                @click="toggleChunk(chunkId)"
              >
                {{ chunkId }}
              </button>
              <span v-if="referenceImageRefs(reference).length" class="evidence-label">可核验图片</span>
              <button
                v-for="imageRef in referenceImageRefs(reference)"
                :key="`${item.case_id}-${imageRef.image_id}`"
                class="evidence-button"
                type="button"
                @click="toggleImage(imageRef, item)"
              >
                {{ imageRef.label || imageRef.display_no || imageRef.image_id }}
              </button>
            </div>
            <div v-for="chunkId in referenceChunkIds(reference)" :key="`${item.case_id}-${chunkId}-detail`" class="chunk-detail">
              <div v-if="expandedChunkId === chunkId" class="detail-box">
                <span v-if="loadingChunkId === chunkId">正在读取文字证据…</span>
                <span v-else-if="chunkErrors[chunkId]" class="error-text">{{ chunkErrors[chunkId] }}</span>
                <span v-else>{{ chunkDetails[chunkId] || "暂无文字证据正文" }}</span>
              </div>
            </div>
            <div v-if="expandedImageId && expandedImageId.scope === 'reference' && expandedImageId.caseId === item.case_id && referenceImageIds(reference).includes(expandedImageId.imageId)" class="image-detail">
              <img :src="assetUrl(expandedImageId.url)" :alt="expandedImageId.caption || expandedImageId.imageId" />
              <span>{{ expandedImageId.caption || expandedImageId.imageId }}</span>
            </div>
          </article>
        </div>
        <p v-else class="empty">暂无</p>
        <h4>关键差异</h4><InfoList :values="item.key_differences" />
        <details v-if="supplementalImages(item).length" class="supplemental-evidence">
          <summary>个例其他图像资料（{{ supplementalImages(item).length }}）</summary>
          <div class="supplemental-actions">
          <button
            v-for="image in supplementalImages(item)"
            :key="`${item.case_id}-supplemental-${image.image_id}`"
            class="evidence-button"
            type="button"
            @click="toggleImage({ image_id: image.image_id, label: image.display_no, caption: image.caption, url: image.url }, item)"
          >
            {{ image.display_no || image.caption || "图片资料" }}
          </button>
          </div>
          <div v-if="expandedImageId && expandedImageId.scope === 'supplemental' && expandedImageId.caseId === item.case_id" class="image-detail">
            <img :src="assetUrl(expandedImageId.url)" :alt="expandedImageId.caption || expandedImageId.imageId" />
            <span>{{ expandedImageId.caption || expandedImageId.imageId }}</span>
          </div>
        </details>
      </article>
    </div>

    <section v-if="hasForecast" class="forecast">
      <header class="forecast-head"><h3>综合研判</h3><span v-if="supportCaseIds.length > 1">跨个例综合</span></header>
      <section v-if="hasForecastSummary" class="core-summary">
        <h4>核心结论摘要</h4>
        <p v-if="forecastSummary.similarity_assessment"><b>判断</b>{{ forecastSummary.similarity_assessment }}</p>
        <p v-for="feature in forecastSummary.core_features || []" :key="feature"><b>关键依据</b>{{ feature }}</p>
        <p v-if="forecastSummary.main_risk" class="risk"><b>最大风险</b>{{ forecastSummary.main_risk }}</p>
        <small v-if="forecastSummary.confidence !== undefined">
          摘要置信度 {{ percent(forecastSummary.confidence) }}<template v-if="supportCaseIds.length"> · 支持个例 {{ supportCaseIds.join("、") }}</template>
        </small>
      </section>
      <ol v-if="result.forecast_tips?.length" class="forecast-tips">
        <li v-for="tip in result.forecast_tips" :key="tip.priority || tip.text">
          <p v-if="tip.focus_object"><b>关注</b>{{ tip.focus_object }}</p>
          <p v-if="tip.possible_bias"><b>偏差</b>{{ tip.possible_bias }}</p>
          <p v-if="tip.suggested_action"><b>建议</b>{{ tip.suggested_action }}</p>
          <p v-if="tip.text && !tip.focus_object">{{ tip.text }}</p>
          <small v-if="tip.confidence !== undefined">{{ tip.consensus_level }} · 置信度 {{ percent(tip.confidence) }}</small>
        </li>
      </ol>
    </section>
  </section>
</template>

<script setup>
import { computed, defineComponent, h, reactive, ref } from "vue";
import { API_ORIGIN } from "../api";

const props = defineProps({ result: { type: Object, required: true } });
const queryTitle = computed(() => String(props.result.query_summary?.process_name || "").trim());
const queryValues = computed(() => {
  const query = props.result.query_summary || {};
  const dateValue = query.date_text || query.date_range || [query.start_date, query.end_date].filter(Boolean).join(" 至 ");
  return [dateValue, ...(query.disaster_types || []), ...(query.affected_areas || []), ...(query.areas || []), ...(query.cities || [])].filter(Boolean);
});
const queryDetail = computed(() => {
  const query = props.result.query_summary || {};
  return [query.observation_description, query.circulation_description, query.intensity_description].filter(Boolean).join("；");
});
const forecastSummary = computed(() => props.result.forecast_summary || {});
const supportCaseIds = computed(() => forecastSummary.value.support_case_ids || []);
const hasForecastSummary = computed(() => Boolean(
  forecastSummary.value.similarity_assessment
  || forecastSummary.value.main_risk
  || forecastSummary.value.core_features?.length,
));
const hasForecast = computed(() => hasForecastSummary.value || Boolean(props.result.forecast_tips?.length));

const expandedChunkId = ref("");
const loadingChunkId = ref("");
const chunkDetails = reactive({});
const chunkErrors = reactive({});
const expandedImageId = ref(null);

// 列表项可能是纯文本，也可能包含带证据编号的结构化参考经验。
const InfoList = defineComponent({
  props: { values: { type: Array, default: () => [] } },
  setup(innerProps) {
    return () => innerProps.values?.length
      ? h("ul", { class: "info-list" }, innerProps.values.map((value) => h("li", typeof value === "object" ? value.text : value)))
      : h("p", { class: "empty" }, "暂无");
  },
});

const percent = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const scoreItems = (item) => {
  const score = item.score_breakdown || {};
  const values = [
    { label: "综合", value: item.retrieval_score }, { label: "落区", value: score.area },
    { label: "灾种", value: score.disaster }, { label: "季节", value: score.temporal },
    { label: "语义", value: score.semantic },
  ];
  for (const [label, value] of Object.entries(item.metric_scores || {})) values.push({ label: `指标·${label}`, value });
  return values.filter((entry) => entry.value !== undefined && entry.value !== null);
};
const referenceText = (value) => typeof value === "object" ? value?.text || "" : String(value || "");
const referenceChunkIds = (value) => typeof value === "object" ? value?.evidence_chunk_ids || [] : [];
const referenceImageRefs = (value) => typeof value === "object" ? value?.evidence_image_refs || [] : [];
const referenceImageIds = (value) => referenceImageRefs(value).map((image) => String(image.image_id || ""));

// 补充资料与主要图片证据去重，只保留没有绑定到参考经验的独立图片。
const supplementalImages = (item) => {
  const primaryIds = new Set([
    ...(item.evidence_images || []).map((image) => String(image.image_id || "")),
    ...(item.key_references || []).flatMap((reference) => referenceImageIds(reference)),
  ]);
  const seen = new Set();
  return (item.supplemental_images || []).filter((image) => {
    const imageId = String(image.image_id || "");
    if (!imageId || primaryIds.has(imageId) || seen.has(imageId)) return false;
    seen.add(imageId);
    return true;
  });
};

// 文字证据按需加载，避免首次渲染把所有 chunk 正文塞进页面。
const toggleChunk = async (chunkId) => {
  if (!chunkId) return;
  if (expandedChunkId.value === chunkId) {
    expandedChunkId.value = "";
    return;
  }
  expandedChunkId.value = chunkId;
  if (chunkDetails[chunkId] || chunkErrors[chunkId]) return;
  loadingChunkId.value = chunkId;
  try {
    const response = await fetch(`${API_ORIGIN}/api/smart-case-match/assets/chunks/${encodeURIComponent(chunkId)}`);
    if (!response.ok) throw new Error(`证据读取失败（${response.status}）`);
    const data = await response.json();
    chunkDetails[chunkId] = data.content || data.text || "暂无文字证据正文";
  } catch (error) {
    chunkErrors[chunkId] = error.message || "文字证据读取失败";
  } finally {
    loadingChunkId.value = "";
  }
};

// 图片同样只在用户点击证据编号时展开，并按当前个例保留关联关系。
const toggleImage = (imageRef, item) => {
  const imageId = String(imageRef?.image_id || "");
  if (!imageId) return;
  if (expandedImageId.value?.imageId === imageId && expandedImageId.value?.caseId === item.case_id) {
    expandedImageId.value = null;
    return;
  }
  const image = [...(item.evidence_images || []), ...(item.supplemental_images || [])]
    .find((candidate) => String(candidate.image_id || "") === imageId);
  expandedImageId.value = {
    scope: (item.key_references || []).some((reference) => referenceImageIds(reference).includes(imageId))
      ? "reference"
      : "supplemental",
    caseId: item.case_id,
    imageId,
    url: image?.url || `/api/smart-case-match/assets/images/${encodeURIComponent(imageId)}`,
    caption: imageRef.caption || image?.caption || imageId,
  };
};
const assetUrl = (url) => /^https?:\/\//i.test(url || "") ? url : `${API_ORIGIN}${String(url || "").startsWith("/") ? url : `/${url}`}`;
</script>

<style scoped>
.smart-result { display: grid; gap: 14px; margin-top: 14px; white-space: normal; }
.query-overview { border-bottom: 1px solid #deded9; display: grid; gap: 8px; padding-bottom: 12px; }
.query-overview > strong { color: #292927; font-size: 14px; line-height: 1.55; }
.query-chips, .tags, .scores { display: flex; flex-wrap: wrap; gap: 6px; }
.query-chips span, .tags span { background: #edf2ef; border: 1px solid #ccd9d3; color: #365f52; font-size: 11px; padding: 3px 7px; }
.query-detail { color: #656560; font-size: 12px; line-height: 1.65; margin: 0; }
.forecast h3 { font-size: 16px; margin: 0; }
.forecast p { margin: 5px 0; }
.forecast b { color: #0d6172; display: inline-block; font-size: 11px; margin-right: 8px; min-width: 58px; }
.case-list { display: grid; gap: 10px; }
.case-card { background: #fff; border: 1px solid #deded9; border-radius: 7px; padding: 12px; }
.case-card header { align-items: flex-start; display: flex; gap: 10px; }
.case-card header h3 { flex: 1; font-size: 14px; margin: 0; }
.case-card header strong { color: #167052; font-size: 15px; }
.meta, .empty, small { color: #777771; font-size: 11px; }
.meta { margin: 5px 0 8px; }
.scores { margin-top: 8px; }
.scores span { background: #f0f0ed; color: #4e4e49; font-size: 11px; padding: 4px 7px; }
h4 { font-size: 12px; margin: 12px 0 4px; }
:deep(.info-list) { margin: 0; padding-left: 19px; }
:deep(.info-list li) { margin: 3px 0; }
.reference-list { display: grid; gap: 0; }
.reference-item { border-bottom: 1px solid #e3e3df; padding: 9px 0; }
.reference-item:last-child { border-bottom: 0; }
.reference-text { align-items: flex-start; display: flex; gap: 8px; margin: 0 0 7px; }
.reference-marker { background: #0d6172; border-radius: 3px; color: #fff; display: inline-grid; flex: 0 0 20px; font-size: 11px; height: 20px; place-items: center; }
.evidence-actions, .supplemental-evidence { display: flex; flex-wrap: wrap; gap: 6px; }
.evidence-label { color: #777771; font-size: 11px; line-height: 25px; }
.evidence-button { background: #fff; border: 1px solid #a8bdb5; border-radius: 4px; color: #365f52; cursor: pointer; font-size: 11px; padding: 4px 7px; }
.evidence-button:hover { background: #edf2ef; }
.detail-box { background: #fff; border-left: 3px solid #7d9d91; color: #4d4d49; font-size: 12px; line-height: 1.7; margin-top: 6px; padding: 7px 9px; white-space: pre-wrap; }
.error-text { color: #b42318; }
.image-detail { margin-top: 8px; }
.image-detail img { border: 1px solid #deded9; max-height: 340px; max-width: 100%; object-fit: contain; }
.image-detail span { color: #656560; display: block; font-size: 11px; margin-top: 3px; }
.supplemental-evidence { display: block; margin-top: 10px; }
.supplemental-evidence summary { color: #656560; cursor: pointer; font-size: 12px; }
.supplemental-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.forecast { border-top: 1px solid #deded9; padding-top: 12px; }
.forecast-head { align-items: center; display: flex; justify-content: space-between; margin-bottom: 8px; }
.forecast-head span { color: #656560; font-size: 11px; }
.core-summary { border-left: 2px solid #0d6172; margin-bottom: 12px; padding: 9px 12px; }
.core-summary h4 { font-size: 13px; margin: 0 0 7px; }
.core-summary .risk { color: #8b2c24; }
.forecast-tips { counter-reset: forecast-tip; display: grid; gap: 8px; list-style: none; margin: 0; padding: 0; }
.forecast-tips li { border: 1px solid #d8dedb; border-radius: 5px; padding: 9px 10px 9px 44px; position: relative; }
.forecast-tips li::before { background: #0d6172; border-radius: 4px; color: #fff; content: counter(forecast-tip); counter-increment: forecast-tip; display: grid; font-size: 11px; height: 22px; left: 10px; place-items: center; position: absolute; top: 10px; width: 22px; }
</style>
