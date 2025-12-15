## Areas Requiring Clarification

The following variables and preferences need to be defined to tailor this plan to your specific needs:

### 1. **Entity Text Matching Strategy**

**Current assumption:** Case-insensitive, whitespace-normalized exact matching.

**Why it matters:** The BioRED paper evaluates using database identifiers (concept IDs), but your requirement specifies text-based comparison. More sophisticated matching (fuzzy, stemming, synonym expansion) would significantly change Phase 3 implementation complexity and accuracy.

**Questions:**
- Should "IL-2" match "interleukin 2" or "IL2"?
- How to handle abbreviations vs. full names?
- What similarity threshold, if any, for fuzzy matching?

### 2. **OpenRouter API Specifics**

**Current assumption:** Standard OpenRouter chat completions endpoint with default rate limits.

**Why it matters:** Rate limits, retry logic, and timeout handling affect reliability.

**Questions:**
- What is the expected request volume? (affects batching strategy)
- Should there be exponential backoff retry logic?
- Any specific OpenRouter tier/plan constraints?

### 3. **Error Handling Philosophy**

**Current assumption:** Log errors and continue processing remaining documents.

**Why it matters:** Affects robustness and debugging.

**Questions:**
- Should the script fail fast on first error?
- Should failed documents be retried automatically?
- How to handle partial API responses?

### 4. **Prompt Tuning Approach**

**Current assumption:** Single static prompt template.

**Why it matters:** Different models may require different prompting strategies.

**Questions:**
- Should the prompt be configurable per model?
- Is there a prompt version tracking requirement?
- Should few-shot examples be included?

### 5. **CSV Location and Permissions**

**Current assumption:** CSV file in current working directory with read/write access.

**Why it matters:** Affects deployment and concurrent usage.

**Questions:**
- Is the CSV shared across multiple users/runs?
- Should file locking be implemented for concurrent access?
- Any specific file path requirements?

### 6. **Progress and Logging**

**Current assumption:** Basic stdout progress reporting.

**Why it matters:** Affects monitoring and debugging in production.

**Questions:**
- Should there be structured logging (JSON format)?
- Is there a specific log level requirement?
- Should progress be saved for resumable runs?

### 7. **Test Data Availability**

**Current assumption:** Tests use synthetic mock data.

**Why it matters:** Integration tests with real BioRED data would be more reliable.

**Questions:**
- Is there a test subset of BioRED data to use?
- Should integration tests call the real OpenRouter API?
- What is the acceptable test execution time?

### 8. **Relation Type Normalization**

**Current assumption:** Exact string matching for relation types (e.g., "Positive_Correlation").

**Why it matters:** LLMs may produce variations like "PositiveCorrelation" or "positive correlation".

**Questions:**
- Should relation types be normalized (lowercase, no underscores)?
- Is there a mapping table for common variations?

### 9. **Multi-Passage Documents**

**Current assumption:** All passages in a document are concatenated with a single space separator.

**Why it matters:** Sentence boundary detection and cross-sentence relations depend on this.

**Questions:**
- What separator between passages? (space, newline, paragraph break)
- Should passage offsets be preserved for debugging?

### 10. **Evaluation Scope**

**Current assumption:** Evaluate all 8 relation types equally.

**Why it matters:** Some relation types are much rarer (e.g., Variant-Variant at 0%).

**Questions:**
- Should there be per-relation-type breakdowns in output?
- Are any relation types to be excluded from evaluation?
- Should the CSV include per-relation-type columns?

