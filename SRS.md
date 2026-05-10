# Software Requirements Specification (SRS)

## EmailGuard — Email Classification & Misinformation Detection System

| Field | Value |
|---|---|
| **Document Version** | 1.0 |
| **Date** | 2026-05-10 |
| **Project Name** | EmailGuard — Email Classification & Misinformation Detection System |
| **Repository** | `DuongD0/Email_Classifying_System` |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture](#3-system-architecture)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Data Requirements](#6-data-requirements)
7. [External Interface Requirements](#7-external-interface-requirements)
8. [System Constraints & Assumptions](#8-system-constraints--assumptions)
9. [Appendices](#9-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for the **EmailGuard** system — an email classification and misinformation detection platform. The system combines a Python-based machine learning pipeline with a React web application to classify emails as **Normal**, **Spam**, or **Fraud**, and to provide users with real-time email security monitoring through Gmail integration.

### 1.2 Scope

The system consists of two major subsystems:

1. **ML Classification Pipeline (Backend/CLI)** — A Python pipeline that ingests, cleans, and trains models on multi-source email corpora, then exposes classification via CLI and programmatic API.
2. **Misinformation Detection Website (Frontend)** — A React single-page application branded as "EmailGuard" that integrates with Gmail via OAuth 2.0 to fetch, analyze, and visualize email threats in real time.

### 1.3 Definitions & Acronyms

| Term | Definition |
|---|---|
| **Normal** | Legitimate, non-malicious email (label 0) |
| **Spam** | Unsolicited bulk/advertising email (label 1) |
| **Fraud** | Phishing, Nigerian-scam, or other deceptive email (label 2) |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; a numerical text representation |
| **Chi²** | Chi-squared statistical test used for feature selection |
| **LogReg** | Logistic Regression classifier |
| **RF** | Random Forest classifier |
| **SPF** | Sender Policy Framework — email authentication protocol |
| **DKIM** | DomainKeys Identified Mail — email signing protocol |
| **DMARC** | Domain-based Message Authentication, Reporting & Conformance |
| **OAuth 2.0** | Open Authorization standard for third-party access delegation |
| **KV Store** | Key-Value persistent storage (Supabase-backed) |

### 1.4 References

- IEEE 830-1998 — Recommended Practice for Software Requirements Specifications
- scikit-learn documentation: https://scikit-learn.org/
- Gmail API reference: https://developers.google.com/gmail/api
- Supabase documentation: https://supabase.com/docs

---

## 2. Overall Description

### 2.1 Product Perspective

EmailGuard is a standalone system that operates in two modes:

- **Offline mode (ML pipeline)**: Data scientists and developers use Python CLI tools to curate datasets, train models, and evaluate classification performance.
- **Online mode (Web application)**: End-users interact with the EmailGuard web dashboard to monitor their Gmail inbox for threats in real time.

### 2.2 Product Functions

| ID | Function | Description |
|---|---|---|
| F-01 | Dataset Assembly | Merge and normalize emails from 7+ public corpora into a unified CSV dataset |
| F-02 | Text Preprocessing | Clean raw email text (HTML stripping, tokenization, lemmatization, stopword removal) |
| F-03 | Model Training | Build and train an ML pipeline (TF-IDF → Chi² → Classifier) on the assembled dataset |
| F-04 | Model Evaluation | Generate accuracy, precision, recall, F1, and confusion matrix metrics |
| F-05 | Single-Email Prediction | Classify a single email via CLI or Python API, returning label + probabilities |
| F-06 | User Authentication | Allow users to sign in/register for the web application |
| F-07 | Gmail OAuth Integration | Connect user Gmail accounts via OAuth 2.0 for read-only email access |
| F-08 | Email Fetching | Retrieve inbox emails from the Gmail API |
| F-09 | Real-Time Email Analysis | Analyze fetched emails for suspicious keywords, authentication failures, and domain reputation |
| F-10 | Security Dashboard | Display summary metrics (total emails analyzed, threats detected, protection rate) |
| F-11 | Trend Analysis | Visualize email threat trends over time (daily, weekly, monthly) |
| F-12 | Risk Analysis | List and categorize senders by risk level (Fraud / Spam / Normal) |
| F-13 | Detailed Email Analysis | Display per-email analysis with highlighted suspicious words and technical headers |
| F-14 | Demo Mode | Allow users to explore the application with sample data without Gmail connection |

### 2.3 User Classes

| User Class | Description |
|---|---|
| **End-User** | A person who connects their Gmail and monitors email threats via the web dashboard |
| **Demo User** | A visitor who explores the system's features using pre-loaded sample data |
| **Data Scientist / Developer** | A technical user who runs the CLI tools to curate datasets, train, and evaluate models |

### 2.4 Operating Environment

| Component | Technology |
|---|---|
| ML Pipeline Runtime | Python 3.9+ |
| Web Frontend | React 18, Vite 6, TypeScript |
| Backend Functions | Supabase Edge Functions (Deno, Hono framework) |
| Persistent Storage | Supabase (PostgreSQL) via KV Store |
| Email Provider | Gmail API (Google Workspace / personal Gmail) |
| Deployment | Static frontend hosting; Supabase for serverless functions |

### 2.5 Design & Implementation Constraints

- The ML pipeline requires scikit-learn, pandas, numpy, nltk, and joblib.
- Gmail integration requires a Google Cloud project with OAuth 2.0 credentials (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`).
- The Supabase backend requires `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` environment variables.
- Large CSV datasets (>100 MB) are tracked via Git LFS.
- The web application is a single-page application (SPA) and requires a modern browser (Chrome, Firefox, Safari, Edge).

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       EmailGuard System                              │
│                                                                      │
│  ┌───────────────────────┐       ┌──────────────────────────────┐   │
│  │   ML Pipeline (Python)│       │   Web App (React + Supabase) │   │
│  │                       │       │                              │   │
│  │  ┌─────────────────┐  │       │  ┌────────────────────────┐  │   │
│  │  │ Extract_email.py│  │       │  │   Frontend (React SPA) │  │   │
│  │  │ (Data Assembly) │  │       │  │   - Auth Page          │  │   │
│  │  └────────┬────────┘  │       │  │   - Dashboard          │  │   │
│  │           │           │       │  │   - Trend Analysis     │  │   │
│  │  ┌────────▼────────┐  │       │  │   - Risk Analysis      │  │   │
│  │  │text_preprocess- │  │       │  │   - Email Analysis     │  │   │
│  │  │    ing.py       │  │       │  └────────────┬───────────┘  │   │
│  │  └────────┬────────┘  │       │               │              │   │
│  │           │           │       │  ┌────────────▼───────────┐  │   │
│  │  ┌────────▼────────┐  │       │  │ Supabase Edge Function │  │   │
│  │  │Email_Classifica-│  │       │  │ (Hono Server)          │  │   │
│  │  │   tion.py       │  │       │  │  - Gmail OAuth         │  │   │
│  │  └────────┬────────┘  │       │  │  - Email Fetch         │  │   │
│  │           │           │       │  │  - Risk Analysis       │  │   │
│  │  ┌────────▼────────┐  │       │  │  - KV Store            │  │   │
│  │  │ Trained Models  │  │       │  └────────────┬───────────┘  │   │
│  │  │ (.joblib/.pkl)  │  │       │               │              │   │
│  │  └─────────────────┘  │       │        ┌──────▼──────┐       │   │
│  └───────────────────────┘       │        │  Gmail API  │       │   │
│                                  │        └─────────────┘       │   │
│                                  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Breakdown

| Module | File(s) | Responsibility |
|---|---|---|
| Text Preprocessing | `text_preprocessing.py` | HTML stripping, tokenization, lemmatization, stopword removal |
| Dataset Assembly | `Extract_email.py` | Load, normalize, deduplicate, and balance multi-source email corpora |
| ML Training & Prediction | `Email_Classification.py` | Build TF-IDF → Chi² → Classifier pipeline; train, evaluate, and predict |
| Web Frontend | `Misinformation Detection Website/src/` | React SPA with pages for auth, dashboard, trends, risks, email analysis |
| Backend Server | `src/supabase/functions/server/index.tsx` | Hono-based API for Gmail OAuth, email fetching, and analysis |
| KV Store | `src/supabase/functions/server/kv_store.tsx` | Supabase-backed key-value persistence for tokens and analysis results |

---

## 4. Functional Requirements

### 4.1 ML Classification Pipeline

#### FR-01: Dataset Assembly

| Field | Description |
|---|---|
| **ID** | FR-01 |
| **Title** | Multi-Source Dataset Assembly |
| **Description** | The system shall load email corpora from multiple CSV sources (Enron, CEAS_08, Ling-Spam, SpamAssassin, phishing_email, Nazario, Nigerian_Fraud), normalize column schemas, map labels to the unified encoding (0=Normal, 1=Spam, 2=Fraud), apply text cleaning, remove duplicates based on `clean_text`, and optionally balance classes by downsampling to the smallest class. |
| **Inputs** | CSV files in `Datasets/` directory |
| **Outputs** | `combined_emails.csv`, `normal_emails.csv`, `spam_emails.csv`, `fraud_emails.csv` |
| **CLI** | `python Extract_email.py [--balance]` |
| **Priority** | High |

#### FR-02: Text Preprocessing

| Field | Description |
|---|---|
| **ID** | FR-02 |
| **Title** | Email Text Cleaning & Normalization |
| **Description** | The system shall process raw email text through the following pipeline: (1) Unescape HTML entities, (2) Strip HTML tags, (3) Normalize whitespace, (4) Convert to lowercase, (5) Remove non-alphabetic characters, (6) Tokenize by whitespace, (7) Filter tokens ≤2 characters, (8) Remove English stopwords, (9) Lemmatize using WordNet. NLTK resources (`stopwords`, `wordnet`) shall be downloaded automatically on first use. Regex patterns shall be cached via `@lru_cache` for performance. |
| **Inputs** | Raw email text string (may contain HTML) |
| **Outputs** | Cleaned, space-joined token string |
| **Priority** | High |

#### FR-03: Model Training

| Field | Description |
|---|---|
| **ID** | FR-03 |
| **Title** | ML Pipeline Training |
| **Description** | The system shall build a scikit-learn `Pipeline` consisting of: (1) `TfidfVectorizer` (unigrams + bigrams, `min_df=2`, English stopwords, configurable `max_features`), (2) Optionally `SelectKBest(chi2, k=k_best)` for feature selection, (3) A classifier — either `LogisticRegression` (balanced, max_iter=1000, lbfgs solver) or `RandomForestClassifier` (balanced, 300 estimators, all CPU cores). The system shall perform a stratified train/test split (default 80/20) and persist the trained pipeline to both `.joblib` and `.pkl` formats. |
| **CLI** | `python Email_Classification.py [--test-size 0.2] [--max-features N] [--k-best K] [--model logreg\|rf] [--balance]` |
| **Outputs** | `models/email_classifier.joblib`, `models/email_classifier.pkl` |
| **Priority** | High |

#### FR-04: Model Evaluation

| Field | Description |
|---|---|
| **ID** | FR-04 |
| **Title** | Classification Performance Evaluation |
| **Description** | After training, the system shall evaluate the model on the held-out test set and report: (1) Overall accuracy, (2) Macro-averaged precision, recall, and F1-score, (3) Per-class precision, recall, F1, and support for Normal, Spam, and Fraud, (4) Confusion matrix (true labels × predicted labels). |
| **Acceptance Criteria** | Accuracy ≥ 90% on the balanced dataset with default hyperparameters. Latest benchmark: Accuracy = 93.66%, Macro F1 = 93.68%. |
| **Priority** | High |

#### FR-05: Single-Email Prediction

| Field | Description |
|---|---|
| **ID** | FR-05 |
| **Title** | Predict Label for a Single Email |
| **Description** | The system shall load a persisted model pipeline and classify a given email text. The output shall include: (1) Predicted label (0/1/2) and label name, (2) Class probabilities for all three classes, (3) Top contributing terms per class with their TF-IDF × coefficient scores (when using Logistic Regression). |
| **CLI** | `python Email_Classification.py --predict "email text here"` |
| **Programmatic API** | `EmailClassifier().predict(text) → (label, label_name)` |
| **Priority** | High |

---

### 4.2 Web Application — Frontend

#### FR-06: User Authentication

| Field | Description |
|---|---|
| **ID** | FR-06 |
| **Title** | User Login / Registration |
| **Description** | The system shall provide a login/registration form with email and password fields. Social authentication buttons (GitHub, Google, Facebook, Twitter) shall be available. Upon successful authentication, the user is directed to the Gmail connection screen (or directly to the dashboard in demo mode). |
| **Pages** | Auth Page (`auth-page.tsx`) |
| **Priority** | High |

#### FR-07: Demo Mode

| Field | Description |
|---|---|
| **ID** | FR-07 |
| **Title** | Demo Mode Access |
| **Description** | The system shall provide a "Try Demo" button on the login page that allows users to explore the application with pre-loaded sample data without connecting a Gmail account. Demo users shall be able to view the Dashboard, Trend Analysis, Risk Analysis, and a sample Email Analysis with a pre-configured phishing email. |
| **Priority** | Medium |

#### FR-08: Gmail OAuth Connection

| Field | Description |
|---|---|
| **ID** | FR-08 |
| **Title** | Gmail Account Connection via OAuth 2.0 |
| **Description** | After authentication, the system shall prompt the user to connect their Gmail account. The system shall: (1) Initiate an OAuth 2.0 authorization flow via Google's `/o/oauth2/v2/auth` endpoint requesting `gmail.readonly` and `userinfo.email` scopes, (2) Handle the OAuth callback to exchange the authorization code for access and refresh tokens, (3) Store tokens securely in the Supabase KV store, (4) Display connection status in the navigation bar. |
| **Privacy** | The system shall only request read-only access to email content. Emails shall not be permanently stored or shared. |
| **Priority** | High |

#### FR-09: Security Dashboard

| Field | Description |
|---|---|
| **ID** | FR-09 |
| **Title** | Email Security Dashboard |
| **Description** | The dashboard shall display: (1) Key metrics cards — Total Emails Analyzed, Fraud & Spam Detected, Protection Rate, Average Response Time, (2) Overall Email Classification pie chart (Normal/Spam/Fraud distribution), (3) Weekly email activity bar chart, (4) Buzzword analysis table showing suspicious words and their frequency, (5) Gmail connection status alert. |
| **Page** | Dashboard (`dashboard.tsx`) |
| **Priority** | High |

#### FR-10: Trend Analysis

| Field | Description |
|---|---|
| **ID** | FR-10 |
| **Title** | Scam Email Trend Visualization |
| **Description** | The Trends page shall display: (1) Monthly scam count with percentage change, (2) Detection rate metric with improvement trend, (3) Monthly/daily trend line/bar charts for scam emails vs. total emails, (4) Email type distribution breakdown (Normal/Spam/Fraud counts and trends), (5) Time pattern heat chart showing scam activity by hour of day, (6) Configurable time range (Daily / Weekly / Monthly). |
| **Page** | Trend Analysis (`trend-analysis.tsx`) |
| **Priority** | Medium |

#### FR-11: Risk Analysis

| Field | Description |
|---|---|
| **ID** | FR-11 |
| **Title** | Sender Risk Analysis & Management |
| **Description** | The Risk Analysis page shall display sender lists categorized as: (1) **Fraud Senders** — senders classified as fraudulent with confidence scores, email counts, last-seen dates, and blocked status, (2) **Spam Senders** — senders classified as spam, (3) **Normal Senders** — trusted senders. Each sender card shall display email address, domain, classification badge, confidence score, and email count. A search bar and filter controls shall allow filtering by classification type. |
| **Page** | Risk Analysis (`risk-analysis.tsx`) |
| **Priority** | Medium |

#### FR-12: Email Analysis

| Field | Description |
|---|---|
| **ID** | FR-12 |
| **Title** | Detailed Per-Email Analysis |
| **Description** | The Email Analysis page shall: (1) Display the Gmail Integration component showing the user's inbox with analyzed/unanalyzed emails, (2) Upon selecting an email, display: risk score (0–100), classification breakdown (Normal/Spam/Fraud indicators), suspicious word analysis with highlighted positions, email headers and technical analysis (SPF, DKIM, DMARC status, domain age, SSL, reputation), (3) Provide tabbed views: Overview, Word Analysis, Technical Details, (4) In demo mode, automatically load a sample phishing email for analysis demonstration. |
| **Page** | Email Analysis (`email-analysis.tsx`) |
| **Priority** | High |

---

### 4.3 Web Application — Backend (Supabase Edge Functions)

#### FR-13: Gmail Status Check

| Field | Description |
|---|---|
| **ID** | FR-13 |
| **Title** | Check Gmail Connection Status |
| **Endpoint** | `GET /make-server-4ca42d89/gmail/status` |
| **Description** | Returns `{ connected: true/false }` based on whether a valid Gmail access token exists in the KV store. |
| **Priority** | High |

#### FR-14: Gmail Auth Initiation

| Field | Description |
|---|---|
| **ID** | FR-14 |
| **Title** | Initiate Gmail OAuth Flow |
| **Endpoint** | `POST /make-server-4ca42d89/gmail/auth` |
| **Description** | Generates and returns a Google OAuth 2.0 authorization URL with the configured `GOOGLE_CLIENT_ID`, redirect URI, and required scopes (`gmail.readonly`, `userinfo.email`). |
| **Priority** | High |

#### FR-15: Gmail OAuth Callback

| Field | Description |
|---|---|
| **ID** | FR-15 |
| **Title** | Process Gmail OAuth Callback |
| **Endpoint** | `POST /make-server-4ca42d89/gmail/callback` |
| **Description** | Accepts `{ code }` in the request body, exchanges the authorization code for access/refresh tokens via Google's token endpoint, and stores the tokens in the KV store. |
| **Priority** | High |

#### FR-16: Fetch Inbox Emails

| Field | Description |
|---|---|
| **ID** | FR-16 |
| **Title** | Retrieve Gmail Inbox Emails |
| **Endpoint** | `GET /make-server-4ca42d89/gmail/emails` |
| **Description** | Fetches the 20 most recent inbox messages from the Gmail API. For each message, retrieves subject, sender, snippet, date, read status, thread ID. Returns `{ emails: [...] }`. Requires a valid stored access token. |
| **Priority** | High |

#### FR-17: Analyze Email

| Field | Description |
|---|---|
| **ID** | FR-17 |
| **Title** | Analyze a Specific Email for Threats |
| **Endpoint** | `POST /make-server-4ca42d89/gmail/analyze/:emailId` |
| **Description** | Fetches the full email content from Gmail API and performs keyword-based risk analysis. Checks for 14 suspicious keywords (e.g., "urgent", "verify account", "free money"), domain reputation, and email authentication headers (SPF, DKIM). Produces a risk score (0–100) and risk level (low/medium/high). Stores the analysis result in the KV store. |
| **Risk Score Calculation** | +15 per matched keyword, +10 for suspicious domain with keywords, +20 for SPF/DKIM failures. Capped at 100. |
| **Priority** | High |

#### FR-18: Get Email Details

| Field | Description |
|---|---|
| **ID** | FR-18 |
| **Title** | Retrieve Stored Email Analysis |
| **Endpoint** | `GET /make-server-4ca42d89/gmail/email/:emailId` |
| **Description** | Returns the stored analysis result for a previously analyzed email, including the full email headers from Gmail. Returns 404 if the email has not been analyzed. |
| **Priority** | Medium |

#### FR-19: Health Check

| Field | Description |
|---|---|
| **ID** | FR-19 |
| **Title** | Server Health Check |
| **Endpoint** | `GET /make-server-4ca42d89/health` |
| **Description** | Returns `{ status: "ok" }` to verify the server is running. |
| **Priority** | Low |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
|---|---|
| NFR-01 | Single-email prediction (CLI) shall complete in < 1 second after model loading. |
| NFR-02 | The web dashboard shall render within 2 seconds on a broadband connection. |
| NFR-03 | Email analysis (backend) shall return a risk score within 3 seconds per email. |
| NFR-04 | The model training pipeline shall complete on the balanced dataset (~103k rows) within 10 minutes on a modern multi-core machine. |
| NFR-05 | Average response time for real-time email analysis shall be ≤ 0.3 seconds (as displayed on dashboard). |

### 5.2 Accuracy

| ID | Requirement |
|---|---|
| NFR-06 | The classification model shall achieve ≥ 90% macro F1-score on the balanced test set. |
| NFR-07 | Per-class recall for Fraud shall be ≥ 90% to minimize missed fraud emails. |
| NFR-08 | Per-class precision for Normal shall be ≥ 95% to minimize false positives flagging legitimate emails. |

### 5.3 Security

| ID | Requirement |
|---|---|
| NFR-09 | Gmail access tokens shall be stored server-side in the KV store and never exposed to the client. |
| NFR-10 | The system shall request only read-only Gmail permissions (`gmail.readonly`). |
| NFR-11 | Email content shall not be permanently stored; only analysis metadata (risk score, matched keywords, first 1000 characters for the analysis session) shall be retained in the KV store. |
| NFR-12 | CORS shall be enabled for all API routes with configurable origin restrictions. |
| NFR-13 | All API requests shall require an `Authorization` header with a valid Supabase anonymous key. |

### 5.4 Usability

| ID | Requirement |
|---|---|
| NFR-14 | The web application shall be responsive and support desktop and mobile viewports. |
| NFR-15 | Navigation shall include both desktop horizontal menu and mobile overflow menu. |
| NFR-16 | Risk levels shall be visually coded: green (Normal/Low), amber (Spam/Medium), red (Fraud/High). |
| NFR-17 | Suspicious words within email content shall be highlighted with color-coded backgrounds. |

### 5.5 Reliability & Availability

| ID | Requirement |
|---|---|
| NFR-18 | The backend edge functions shall be deployed on Supabase with automatic scaling. |
| NFR-19 | If the Gmail API is unreachable, the system shall display a user-friendly error message and allow retry. |
| NFR-20 | If the ML model file is missing or corrupted, the training CLI shall prompt the user to retrain. |

### 5.6 Maintainability

| ID | Requirement |
|---|---|
| NFR-21 | The ML pipeline shall be modular: text preprocessing, dataset assembly, and classification shall be in separate Python modules. |
| NFR-22 | The web application shall use a component-based architecture with reusable UI components (shadcn/ui pattern). |
| NFR-23 | Backend API routes shall be organized in a single Hono server with clear endpoint naming. |

---

## 6. Data Requirements

### 6.1 Dataset Sources

| Source | Records (approx.) | Labels | Format |
|---|---|---|---|
| Enron.csv | ~30k | 0=Normal, 1=Spam | CSV (subject, body, label) |
| CEAS_08.csv | ~20k | 0=Normal, 1=Spam | CSV (subject, body, label) |
| Ling.csv | ~2.8k | 0=Normal, 1=Spam | CSV |
| SpamAssassin.csv | ~6k | 0=Normal, 1=Spam | CSV |
| phishing_email.csv | ~11k | 0=Normal, 1=Fraud | CSV (text_combined) |
| Nazario.csv | ~5k | All Fraud | CSV |
| Nigerian_Fraud.csv | ~5k | All Fraud | CSV |

### 6.2 Combined Dataset

| Attribute | Value |
|---|---|
| Total records (unbalanced) | ~134,000 |
| Total records (balanced) | ~103,000 (34,385 per class) |
| Columns | `subject`, `body`, `text`, `clean_text`, `label`, `label_name`, `source` |
| Deduplication key | `clean_text` |
| File format | CSV |
| Storage | Git LFS |

### 6.3 Model Artifacts

| Artifact | Format | Purpose |
|---|---|---|
| `models/email_classifier.joblib` | Joblib | Fast model loading (recommended for production) |
| `models/email_classifier.pkl` | Pickle | Interoperability with non-Joblib tools |

### 6.4 KV Store Schema

The Supabase KV store table `kv_store_4ca42d89` stores:

| Key Pattern | Value Type | Description |
|---|---|---|
| `gmail_access_token` | String | Google OAuth 2.0 access token |
| `gmail_refresh_token` | String | Google OAuth 2.0 refresh token |
| `email_analysis_{emailId}` | JSON Object | Analysis result: `{ riskScore, riskLevel, matchedKeywords, analyzedAt, sender, subject, content }` |

---

## 7. External Interface Requirements

### 7.1 User Interfaces

| Interface | Description |
|---|---|
| **Auth Page** | Login/Register form with social auth buttons and Gmail connection prompt. Demo mode entry button. |
| **Navigation Bar** | Fixed top bar with logo ("EmailGuard"), Gmail connection badge, page links (Dashboard, Email Trends, Risk Analysis, Email Analysis), and Logout button. Mobile-responsive with horizontal scroll menu. |
| **Dashboard** | 4 KPI cards, pie chart, bar chart, buzzword table. |
| **Trend Analysis** | Time-range selector, line/bar/area charts, hourly heat chart, email type breakdown. |
| **Risk Analysis** | Search bar, classification filter, sender cards grouped by Fraud/Spam/Normal. |
| **Email Analysis** | Gmail inbox list, email detail panel with tabs (Overview, Word Analysis, Technical), risk score gauge, highlighted content. |

### 7.2 Hardware Interfaces

No special hardware is required. Standard computing devices with internet access.

### 7.3 Software Interfaces

| System | Interface | Protocol |
|---|---|---|
| Gmail API | `googleapis.com/gmail/v1/` | HTTPS / REST |
| Google OAuth 2.0 | `accounts.google.com/o/oauth2/v2/auth`, `oauth2.googleapis.com/token` | HTTPS |
| Supabase | Supabase client SDK | HTTPS |

### 7.4 Communication Interfaces

- All frontend-to-backend communication uses HTTPS REST API calls.
- CORS is configured to accept all origins (`*`) with `Content-Type` and `Authorization` headers.
- The Gmail API uses Bearer token authentication.

---

## 8. System Constraints & Assumptions

### 8.1 Constraints

| ID | Constraint |
|---|---|
| C-01 | Gmail integration requires a Google Cloud project with OAuth credentials. |
| C-02 | The ML pipeline requires Python 3.9+ and the packages listed in `requirements.txt`. |
| C-03 | Large datasets (>100 MB CSV) require Git LFS for version control. |
| C-04 | The Supabase backend requires Deno runtime for edge functions. |
| C-05 | Gmail API rate limits apply (250 quota units per user per second). |
| C-06 | Email content analysis is currently keyword-based on the backend; the ML model is used via CLI. |

### 8.2 Assumptions

| ID | Assumption |
|---|---|
| A-01 | Users have a Gmail account (personal or Workspace) to connect. |
| A-02 | The ML training data is representative of real-world email distributions. |
| A-03 | NLTK resources (stopwords, WordNet) can be downloaded at runtime. |
| A-04 | Users operate the web application on a modern browser with JavaScript enabled. |
| A-05 | The Supabase project is provisioned and the KV store table is created. |

---

## 9. Appendices

### 9.1 ML Pipeline CLI Reference

```bash
# Dataset assembly
python Extract_email.py [--balance]

# Model training
python Email_Classification.py \
  [--test-size 0.2] \
  [--max-features 150000] \
  [--k-best 2000] \
  [--model logreg|rf] \
  [--balance | --no-balance] \
  [--model-path PATH] \
  [--pickle-path PATH]

# Single prediction
python Email_Classification.py --predict "Your email text here"
```

### 9.2 Programmatic Prediction API

```python
from Email_Classification import EmailClassifier

clf = EmailClassifier()
label, label_name = clf.predict("Schedule our standup meeting")
# label = 0, label_name = "Normal"
```

### 9.3 Backend API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| GET | `/make-server-4ca42d89/health` | Health check |
| GET | `/make-server-4ca42d89/gmail/status` | Gmail connection status |
| POST | `/make-server-4ca42d89/gmail/auth` | Initiate Gmail OAuth |
| POST | `/make-server-4ca42d89/gmail/callback` | Process OAuth callback |
| GET | `/make-server-4ca42d89/gmail/emails` | Fetch inbox emails |
| POST | `/make-server-4ca42d89/gmail/analyze/:emailId` | Analyze specific email |
| GET | `/make-server-4ca42d89/gmail/email/:emailId` | Get stored analysis |

### 9.4 Latest Model Benchmark (Balanced Logistic Regression)

| Metric | Value |
|---|---|
| Accuracy | 93.66% |
| Macro Precision | 93.80% |
| Macro Recall | 93.66% |
| Macro F1 | 93.68% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 96.67% | 94.04% | 95.33% | 6,877 |
| Spam | 89.42% | 95.40% | 92.32% | 6,877 |
| Fraud | 95.32% | 91.54% | 93.39% | 6,877 |

**Confusion Matrix:**

|  | Pred Normal | Pred Spam | Pred Fraud |
|---|---|---|---|
| **True Normal** | 6,467 | 283 | 127 |
| **True Spam** | 134 | 6,561 | 182 |
| **True Fraud** | 89 | 493 | 6,295 |

### 9.5 Suspicious Keywords for Web-Based Analysis

The backend risk analysis checks for the following keywords:

`urgent`, `limited time`, `act now`, `click here`, `verify account`, `suspended`, `winner`, `congratulations`, `free money`, `lottery`, `inheritance`, `prince`, `million dollars`, `tax refund`

### 9.6 Technology Stack Summary

| Layer | Technology | Version |
|---|---|---|
| ML Runtime | Python | 3.9+ |
| ML Libraries | scikit-learn, pandas, numpy, nltk, joblib | See `requirements.txt` |
| Frontend Framework | React | 18.3 |
| Build Tool | Vite | 6.3 |
| Language | TypeScript | — |
| UI Components | Radix UI, Lucide Icons, Recharts, Tailwind CSS | — |
| Backend Runtime | Deno (Supabase Edge Functions) | — |
| API Framework | Hono | latest |
| Database | Supabase (PostgreSQL) | — |
| Email API | Gmail API v1 | — |
| Auth | Google OAuth 2.0 | — |
| Version Control | Git + Git LFS | — |
