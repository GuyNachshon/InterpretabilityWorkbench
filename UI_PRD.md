# InterpretabilityWorkbench UI - Product Requirements Document

**Document Version**: 1.0  
**Date**: 2025-01-31  
**Target Audience**: Frontend Development Team  
**Project**: Interactive Mechanistic-Interpretability Workbench for LLMs  

---

## 📋 Executive Summary

The InterpretabilityWorkbench UI is a **React-based web application** that provides researchers with an interactive interface to:
- Browse discovered SAE features with real-time token analysis
- Create and toggle live LoRA patches on language models
- Visualize feature relationships and provenance graphs
- Monitor inference results with <400ms latency

This document serves as the complete handoff specification for frontend implementation.

---

## 🎯 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Patch Toggle Latency** | <400ms | WebSocket round-trip time |
| **Feature Load Time** | <2s | Initial feature table population |
| **UI Responsiveness** | <100ms | Click-to-visual-feedback |
| **Concurrent Users** | 5+ | Multiple researchers simultaneously |

---

## 👥 User Personas & Journeys

### **Primary: Alignment Researcher (Alice)**
- **Goal**: Find and disable dangerous model behaviors
- **Journey**: Load model → Browse features → Identify suspicious patterns → Create suppression patch → Verify safety
- **Key Pain Points**: Need to quickly identify high-risk features, toggle patches rapidly

### **Secondary: ML Engineer (Ben)**
- **Goal**: Export production-ready patches
- **Journey**: Load colleague's work → Review feature analysis → Test patches → Export for deployment
- **Key Pain Points**: Need confidence in patch stability, clear export workflows

---

## 🏗️ Technical Architecture

### **Frontend Stack**
```
React 18+ (with hooks)
├── State Management: Redux Toolkit or Zustand
├── UI Framework: Material-UI v5 or Tailwind CSS
├── Data Visualization: D3.js + Plotly.js
├── WebSocket: Socket.io-client or native WebSocket
├── HTTP Client: Axios or fetch
└── Build Tool: Vite or Create React App
```

### **Backend Integration**
- **Base URL**: `http://localhost:8000` (configurable)
- **WebSocket**: `ws://localhost:8000/ws`
- **API Spec**: FastAPI auto-generated OpenAPI docs at `/docs`

---

## 🖼️ UI Components & Layout

### **1. Main Navigation**
```
┌─ InterpretabilityWorkbench ─────────────────────────────┐
│ [Model Status] [Feature Count] [Active Patches] [⚙️]    │
└─────────────────────────────────────────────────────────┘
```

**Requirements:**
- Always-visible status bar showing current model and SAE loaded
- Real-time counters (features discovered, patches active)
- Settings dropdown for API endpoint configuration

### **2. Feature Explorer (Primary View)**
```
┌─ Features ──────────┬─ Detail Panel ─────────────────────┐
│ 🔍 [Search____]     │ Feature layer_8_feature_42          │
│ [🏷️Layer] [📊Sort]  │                                     │
│                     │ ┌─ Top Tokens ─────────────────────┐ │
│ ┃ F42  sparsity:.15 │ │ "the" (0.89) "a" (0.67) "."    │ │
│ ┃      strength:.82  │ │ "and" (0.54) "in" (0.43)       │ │
│ ┃      [👁️] [🔧]     │ └─────────────────────────────────┘ │
│                     │                                     │
│ ┃ F127 sparsity:.08 │ ┌─ Token Cloud ──────────────────────┐ │
│ ┃      strength:.94  │ │     the                            │ │
│ ┃      [👁️] [🔧]     │ │  a      and    in                  │ │
│                     │ │      .      of                     │ │
│ ┃ F201 sparsity:.22 │ └─────────────────────────────────┘ │
│ ┃      strength:.76  │                                     │
│ ┃      [👁️] [🔧]     │ [Create Patch] [Export Feature]     │
└─────────────────────┴─────────────────────────────────────┘
```

**Requirements:**
- **Feature List** (left panel):
  - Sortable by sparsity, activation strength, feature ID
  - Filterable by layer, activation frequency
  - Search by top tokens
  - Pagination (100 features per page)
  - Icons: 👁️ = view details, 🔧 = create patch

- **Detail Panel** (right panel):
  - Top 10 activating tokens with strengths
  - Interactive token cloud (hover effects)
  - Context snippets showing where tokens appear
  - Quick action buttons

### **3. Live Patching Interface**
```
┌─ Active Patches ────────────────────────────────────────┐
│ patch_layer8_feature42    [🟢 ON ] [-2.0] [❌]         │
│ patch_layer12_feature99   [🔴 OFF] [+1.5] [❌]         │
│                                                         │
│ [+ Create New Patch]                                    │
└─────────────────────────────────────────────────────────┘

┌─ Inference Test ────────────────────────────────────────┐
│ Input: [The AI system should never_____________]        │
│ [▶️ Run Inference]                                      │
│                                                         │
│ Top Predictions:                                        │
│ ████████████ "harm" (0.23) → 0.15 (-35%) ⬇️            │
│ ██████████   "lie"  (0.19) → 0.08 (-58%) ⬇️            │
│ ████████     "help" (0.15) → 0.31 (+107%) ⬆️           │
└─────────────────────────────────────────────────────────┘
```

**Requirements:**
- **Patch Management**:
  - Toggle switches with immediate WebSocket updates
  - Strength sliders (-5.0 to +5.0 range)
  - Delete patch capability
  - Visual status indicators (green=active, red=inactive)

- **Real-time Inference**:
  - Text input field with autocomplete
  - Progress indicator during inference
  - Bar chart showing probability changes
  - Color coding: ⬆️ green (increased), ⬇️ red (decreased)
  - **Latency display**: Show actual response time

### **4. Provenance Graph**
```
┌─ Feature Relationships ─────────────────────────────────┐
│                    F127                                 │
│                   ╱    ╲                               │
│                F42      F201                           │
│               ╱  ╲    ╱     ╲                          │
│            F12   F88 F156   F234                       │
│                                                         │
│ Controls: [🔍 Zoom] [↻ Reset] [📥 Export]              │
│ Layer: [8 ▼] Depth: [2 ▼] Layout: [Force ▼]           │
└─────────────────────────────────────────────────────────┘
```

**Requirements:**
- **D3.js force-directed graph**
- Node size = activation strength
- Edge thickness = relationship strength  
- Interactive: click nodes for details, drag to reposition
- Controls: zoom, pan, reset view
- Layer selector and depth control
- Export as SVG/PNG

### **5. Model Management**
```
┌─ Model & SAE Configuration ─────────────────────────────┐
│ Model: [microsoft/DialoGPT-small ▼] [🔄 Load]          │
│ Status: ✅ Loaded (2.1GB VRAM)                          │
│                                                         │
│ SAE Layer 8: [📁 Browse] [sae_layer_8.safetensors]     │
│ Activations: [📁 Browse] [layer8_activations.parquet]  │
│ Status: ✅ 1,247 features analyzed                      │
│                                                         │
│ [🔄 Reload] [📤 Export All] [📋 View Logs]              │
└─────────────────────────────────────────────────────────┘
```

**Requirements:**
- Model dropdown with HuggingFace model search
- File browser for SAE weights and activation data
- Progress indicators for loading operations
- Memory usage monitoring
- Error handling with clear messages

---

## 🔌 API Integration Specification

### **REST Endpoints**

| Method | Endpoint | Purpose | Response Time |
|--------|----------|---------|---------------|
| `POST` | `/load-model` | Load HF model | <30s |
| `POST` | `/load-sae` | Load SAE + activations | <10s |
| `GET` | `/features?layer=8&limit=100` | Get feature list | <2s |
| `GET` | `/feature/{id}/details` | Get feature details | <500ms |
| `POST` | `/patch` | Create LoRA patch | <1s |
| `POST` | `/patch/{id}/toggle` | Toggle patch | <100ms |
| `POST` | `/inference` | Run inference | <400ms |
| `POST` | `/export-features` | Export data | <5s |

### **WebSocket Events**

#### **Client → Server**
```javascript
// Request inference with current patches
{
  "type": "inference_request",
  "text": "The AI should never",
  "request_id": "uuid-1234"
}

// Subscribe to patch updates
{
  "type": "subscription",
  "subscription_type": "patch_updates",
  "action": "subscribe"
}
```

#### **Server → Client**
```javascript
// Inference results
{
  "type": "inference_result",
  "data": {
    "input_text": "The AI should never",
    "token_probabilities": [
      {"token": "harm", "probability": 0.15, "logit": -1.2},
      {"token": "help", "probability": 0.31, "logit": 0.8}
    ],
    "latency_ms": 347
  },
  "request_id": "uuid-1234"
}

// Patch toggled by another user
{
  "type": "patch_toggled",
  "patch_id": "patch_layer8_feature42",
  "action": "enabled",
  "timestamp": 1643723400
}
```

---

## 🎨 Design System

### **Color Palette**
```css
/* Primary */
--primary-50:  #f0f7ff;
--primary-500: #3b82f6;  /* Main brand blue */
--primary-700: #1d4ed8;

/* Semantic */
--success: #10b981;      /* Feature active, patch enabled */
--warning: #f59e0b;      /* Loading, processing */
--danger:  #ef4444;      /* High-risk features, errors */
--info:    #6366f1;      /* Neutral information */

/* Activation Heatmap */
--activation-0: #f8fafc;  /* No activation */
--activation-1: #ddd6fe;  /* Low activation */
--activation-2: #8b5cf6;  /* Medium activation */  
--activation-3: #7c3aed;  /* High activation */
--activation-4: #5b21b6;  /* Very high activation */
```

### **Typography**
```css
/* Headings */
h1: Inter 24px/32px 600
h2: Inter 20px/28px 600  
h3: Inter 16px/24px 600

/* Body */
body: Inter 14px/20px 400
small: Inter 12px/16px 400

/* Code/Data */
code: 'JetBrains Mono' 13px/18px 400
```

### **Spacing System**
```css
/* 4px base unit */
xs: 4px   /* 1 unit */
sm: 8px   /* 2 units */
md: 16px  /* 4 units */
lg: 24px  /* 6 units */
xl: 32px  /* 8 units */
```

---

## 📱 Responsive Design Requirements

### **Desktop Primary (1920x1080)**
- Three-column layout: Features | Detail | Patches
- Full provenance graph visibility
- All controls accessible

### **Laptop (1366x768)**
- Two-column layout: Features + Detail (tabbed)
- Collapsible panels
- Simplified graph view

### **Tablet (768x1024) - Optional**
- Single column with drawer navigation
- Touch-optimized controls
- Simplified feature list

---

## ⚡ Performance Requirements

### **Bundle Size**
- Initial JS bundle: <2MB gzipped
- Code splitting by route
- Lazy load D3.js visualization

### **Runtime Performance**
- Feature list virtualization (100+ items)
- Debounced search (300ms)
- WebSocket connection pooling
- Memory cleanup on component unmount

### **Accessibility (WCAG 2.1 AA)**
- Keyboard navigation for all controls
- Screen reader support for feature data
- High contrast mode compatibility
- Focus management for modals

---

## 🧪 Testing Requirements

### **Unit Tests**
- [ ] Component rendering
- [ ] WebSocket message handling  
- [ ] State management actions
- [ ] API error handling

### **Integration Tests**
- [ ] Feature loading workflow
- [ ] Patch creation and toggle
- [ ] Real-time inference updates
- [ ] Export functionality

### **E2E Tests (Cypress/Playwright)**
- [ ] Complete researcher workflow
- [ ] Multi-user patch collaboration
- [ ] Model loading and SAE analysis
- [ ] Performance under load

---

## 📦 Deployment & Environment

### **Development**
```bash
npm install && npm run dev
# Connects to localhost:8000 backend
```

### **Production Build**
```bash
npm run build
# Static files served by FastAPI at /
# Environment variables:
# - REACT_APP_API_URL=https://your-api.com
# - REACT_APP_WS_URL=wss://your-api.com/ws
```

### **Docker Support**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

---

## 🚨 Error Handling

### **Error States**
1. **Model Loading Failed**: Show retry button with error details
2. **WebSocket Disconnected**: Auto-reconnect with status indicator
3. **Patch Creation Failed**: Clear error message with suggested fixes
4. **Large Dataset Timeout**: Progressive loading with cancel option

### **Loading States**
- Skeleton screens for feature lists
- Progress bars for model loading
- Spinner overlays for inference requests
- Pulse animations for real-time updates

---

## 🔐 Security Considerations

### **Data Protection**
- No activation data leaves the browser
- API tokens in secure HTTP-only cookies
- Input sanitization for text inference
- CSP headers for XSS protection

### **Model Safety**
- Warning dialogs for destructive patches
- Confirmation for high-strength modifications  
- Audit log of patch modifications
- Emergency "disable all patches" button

---

## 📈 Analytics & Monitoring

### **User Metrics**
- Feature exploration patterns
- Patch creation frequency
- Inference request volume
- Session duration

### **Performance Metrics**
- API response times
- WebSocket latency
- Frontend bundle performance
- Error rates by component

---

## 🚀 Implementation Phases

### **Phase 1: Core MVP (2 weeks)**
- [ ] Basic feature list and details
- [ ] Model loading interface
- [ ] Simple patch creation
- [ ] WebSocket integration

### **Phase 2: Advanced Features (2 weeks)**
- [ ] Interactive token clouds
- [ ] Real-time inference testing
- [ ] Provenance graph visualization
- [ ] Export functionality

### **Phase 3: Polish & Performance (1 week)**
- [ ] Responsive design
- [ ] Performance optimization
- [ ] Error handling
- [ ] Testing coverage

---

## 📋 Acceptance Criteria

### **Functional**
- [ ] Load model and SAE within 30 seconds
- [ ] Browse 1000+ features smoothly
- [ ] Create and toggle patches with <400ms latency
- [ ] Real-time inference updates via WebSocket
- [ ] Export features and patches successfully

### **Non-Functional**
- [ ] Works in Chrome, Firefox, Safari latest
- [ ] Accessible via keyboard navigation
- [ ] Responsive down to 1366x768
- [ ] <2s initial page load
- [ ] No memory leaks during extended use

---

## 🤝 Handoff Checklist

### **Backend Team Provides**
- [ ] ✅ FastAPI server running on localhost:8000
- [ ] ✅ WebSocket endpoint at /ws
- [ ] ✅ OpenAPI documentation at /docs
- [ ] ✅ Sample SAE models and activation data
- [ ] ✅ CORS configuration for development

### **Frontend Team Delivers**
- [ ] React application with all specified features
- [ ] Integration with provided API endpoints
- [ ] Responsive design implementation
- [ ] Test coverage >80%
- [ ] Documentation for future maintenance

---

## 📞 Support & Communication

**Backend API Contact**: Available for integration questions  
**Design Questions**: Reference this PRD, Material-UI design system  
**Performance Issues**: Target metrics defined in Success Criteria  
**Timeline Concerns**: 5-week estimate with 3 defined phases  

---

**Ready to build an amazing interpretability tool! 🚀**

*This PRD serves as the complete specification for frontend development. All backend APIs are implemented and tested. The frontend team has everything needed to create an intuitive, performant UI for cutting-edge AI safety research.*