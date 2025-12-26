# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create backend web module directory structure
  - Create frontend React project with Vite
  - Install backend dependencies (FastAPI, python-socketio, uvicorn)
  - Install frontend dependencies (React, TypeScript, Ant Design, Axios, Socket.IO client, i18next)
  - Set up development environment configuration
  - _Requirements: 9.1_

- [x] 1.1 Write unit tests for project setup
  - Test backend module imports
  - Test frontend build configuration
  - _Requirements: 9.1_

- [x] 2. Implement backend API server foundation
  - [x] 2.1 Create FastAPI application with CORS middleware
    - Implement `ollaforge/web/server.py` with FastAPI app initialization
    - Configure CORS for development and production
    - Set up Socket.IO server integration
    - _Requirements: 9.1, 9.3_

  - [x] 2.2 Define API request/response models
    - Create `ollaforge/web/models.py` with Pydantic models
    - Implement GenerationRequest, AugmentationRequest, TaskStatus models
    - Add validation rules for all models
    - _Requirements: 9.2_

  - [x] 2.3 Write property test for API models
    - **Property 30: API uses JSON format**
    - **Validates: Requirements 9.2**

  - [x] 2.4 Write unit tests for CORS configuration
    - Test CORS headers in responses
    - Test allowed origins configuration
    - _Requirements: 9.3_

  - [x] 2.5 Write property test for CORS headers
    - **Property 31: CORS headers are present**
    - **Validates: Requirements 9.3**

- [x] 3. Implement generation service and routes
  - [x] 3.1 Create generation service wrapper
    - Implement `ollaforge/web/services/generation.py`
    - Wrap existing `generate_data_concurrent` with async interface
    - Add progress callback support
    - _Requirements: 1.2, 3.1, 3.2_

  - [x] 3.2 Implement generation API routes
    - Create `ollaforge/web/routes/generation.py`
    - Implement POST `/api/generate` endpoint
    - Implement GET `/api/generate/{task_id}` status endpoint
    - Implement GET `/api/generate/{task_id}/download` endpoint
    - _Requirements: 1.2, 1.3_

  - [x] 3.3 Write property test for generation initiation
    - **Property 1: Valid generation parameters initiate processing**
    - **Validates: Requirements 1.2**

  - [x] 3.4 Write property test for generation download
    - **Property 2: Completed generation provides download**
    - **Validates: Requirements 1.3**

  - [x] 3.5 Write property test for generation errors
    - **Property 3: Generation failures display errors**
    - **Validates: Requirements 1.4**

  - [x] 3.6 Write unit tests for generation routes
    - Test route parameter validation
    - Test error responses
    - _Requirements: 1.2, 1.3, 1.4_

- [x] 4. Implement augmentation service and routes
  - [x] 4.1 Create augmentation service wrapper
    - Implement `ollaforge/web/services/augmentation.py`
    - Wrap existing `DatasetAugmentor` with async interface
    - Add preview functionality
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 4.2 Implement augmentation API routes
    - Create `ollaforge/web/routes/augmentation.py`
    - Implement POST `/api/augment/upload` endpoint
    - Implement POST `/api/augment/preview` endpoint
    - Implement POST `/api/augment` endpoint
    - Implement GET `/api/augment/{task_id}/download` endpoint
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 4.3 Write property test for file upload
    - **Property 4: File upload extracts fields**
    - **Validates: Requirements 2.1**

  - [x] 4.4 Write property test for field validation
    - **Property 5: Field validation works correctly**
    - **Validates: Requirements 2.2**

  - [x] 4.5 Write property test for augmentation download
    - **Property 6: Completed augmentation provides download**
    - **Validates: Requirements 2.4**

  - [x] 4.6 Write property test for partial failures
    - **Property 7: Partial augmentation failures preserve data**
    - **Validates: Requirements 2.5**

  - [x] 4.7 Write unit tests for augmentation routes
    - Test file upload handling
    - Test preview generation
    - Test error responses
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 5. Implement WebSocket progress tracking
  - [x] 5.1 Create WebSocket event handlers
    - Implement `ollaforge/web/routes/websocket.py`
    - Handle connect/disconnect events
    - Implement task subscription mechanism
    - _Requirements: 3.1, 3.2_

  - [x] 5.2 Integrate progress callbacks with WebSocket
    - Emit progress events during generation
    - Emit progress events during augmentation
    - Handle error events
    - _Requirements: 3.1, 3.2, 3.4_

  - [x] 5.3 Write property test for progress display
    - **Property 8: Progress indicators show during operations**
    - **Validates: Requirements 3.1**

  - [x] 5.4 Write property test for real-time updates
    - **Property 9: Progress updates in real-time**
    - **Validates: Requirements 3.2**

  - [x] 5.5 Write property test for completion statistics
    - **Property 10: Completion shows statistics**
    - **Validates: Requirements 3.3**

  - [x] 5.6 Write property test for error handling
    - **Property 11: Errors don't stop progress display**
    - **Validates: Requirements 3.4**

  - [x] 5.7 Write unit tests for WebSocket handlers
    - Test connection handling
    - Test subscription mechanism
    - Test event emission
    - _Requirements: 3.1, 3.2, 3.4_

- [x] 6. Implement model management routes
  - [x] 6.1 Create model API routes
    - Implement `ollaforge/web/routes/models.py`
    - Implement GET `/api/models` endpoint using existing `get_available_models`
    - Implement GET `/api/models/{model_name}` endpoint
    - Add error handling for Ollama unavailability
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x]* 6.2 Write property test for model information
    - **Property 33: Model information includes size**
    - **Validates: Requirements 10.2**

  - [x]* 6.3 Write property test for model validation
    - **Property 34: Model validation before generation**
    - **Validates: Requirements 10.4**

  - [x]* 6.4 Write property test for Ollama errors
    - **Property 25: Ollama unavailability shows clear error**
    - **Validates: Requirements 7.5**

  - [x]* 6.5 Write unit tests for model routes
    - Test model list retrieval
    - Test model info retrieval
    - Test error handling
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 7. Implement multi-format file support
  - [x] 7.1 Extend file upload to support all formats
    - Add format detection for uploaded files
    - Support JSONL, JSON, CSV, TSV, Parquet
    - _Requirements: 4.1_

  - [x] 7.2 Implement format conversion for downloads
    - Add format parameter to download endpoints
    - Convert datasets to requested format
    - _Requirements: 4.2, 4.3_

  - [x]* 7.3 Write property test for format support
    - **Property 12: Format support is comprehensive**
    - **Validates: Requirements 4.1**

  - [x]* 7.4 Write property test for format conversion
    - **Property 13: Format conversion preserves data**
    - **Validates: Requirements 4.3**

  - [x]* 7.5 Write property test for unsupported formats
    - **Property 14: Unsupported formats show errors**
    - **Validates: Requirements 4.4**

  - [x]* 7.6 Write unit tests for format handling
    - Test format detection
    - Test format conversion
    - Test error handling
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 8. Implement task management and concurrency
  - [x] 8.1 Create task manager for concurrent requests
    - Implement task queue and status tracking
    - Handle multiple concurrent requests
    - Implement request queueing for resource limits
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 8.2 Add timeout handling
    - Implement request timeout mechanism
    - Return appropriate error responses
    - _Requirements: 7.4_

  - [x]* 8.3 Write property test for concurrent requests
    - **Property 21: Concurrent requests are independent**
    - **Validates: Requirements 7.1**

  - [x]* 8.4 Write property test for non-blocking operations
    - **Property 22: Operations don't block endpoints**
    - **Validates: Requirements 7.2**

  - [x]* 8.5 Write property test for request queueing
    - **Property 23: Resource limits trigger queueing**
    - **Validates: Requirements 7.3**

  - [x]* 8.6 Write property test for timeout handling
    - **Property 24: Timeouts return error responses**
    - **Validates: Requirements 7.4**

  - [x]* 8.7 Write unit tests for task manager
    - Test task creation and tracking
    - Test concurrent execution
    - Test queueing behavior
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 9. Checkpoint - Backend API complete
  - Ensure all backend tests pass
  - Verify API endpoints are functional
  - Test WebSocket communication
  - Ask the user if questions arise

- [x] 10. Set up frontend React application structure
  - [x] 10.1 Configure React Router for navigation
    - Set up routing with React Router
    - Create page layout structure
    - _Requirements: 9.1_

  - [x] 10.2 Configure Ant Design theme
    - Configure theme
    - Set up global styles
    - _Requirements: 1.1, 2.1_

  - [x] 10.3 Set up internationalization
    - Configure i18next with language detection
    - Create translation files for English and Chinese
    - Implement language detection
    - _Requirements: 8.1, 8.2_

  - [ ]* 10.4 Write unit tests for i18n setup
    - Test language detection
    - Test translation loading
    - _Requirements: 8.1_

- [x] 11. Implement API client services
  - [x] 11.1 Create Axios API client
    - Implement `src/services/api.ts`
    - Configure base URL and headers
    - Add request/response interceptors
    - _Requirements: 9.2_

  - [x] 11.2 Implement generation API methods
    - Create `generationAPI` with startGeneration, getStatus, downloadDataset
    - Add error handling
    - _Requirements: 1.2, 1.3_

  - [x] 11.3 Implement augmentation API methods
    - Create `augmentationAPI` with uploadDataset, previewAugmentation, startAugmentation
    - Add file upload handling
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 11.4 Implement models API methods
    - Create `modelsAPI` with listModels, getModelInfo
    - _Requirements: 10.1, 10.2_

  - [ ]* 11.5 Write unit tests for API client
    - Test API method calls
    - Test error handling
    - Test request formatting
    - _Requirements: 9.2_

- [x] 12. Implement WebSocket client
  - [x] 12.1 Create Socket.IO client wrapper
    - Implement `src/services/websocket.ts`
    - Handle connection/disconnection
    - Implement task subscription
    - _Requirements: 3.1, 3.2_

  - [x] 12.2 Add reconnection logic
    - Handle connection failures
    - Implement automatic reconnection
    - _Requirements: 3.1_

  - [ ]* 12.3 Write unit tests for WebSocket client
    - Test connection handling
    - Test subscription mechanism
    - Test reconnection logic
    - _Requirements: 3.1, 3.2_

- [x] 13. Implement shared UI components
  - [x] 13.1 Create ProgressDisplay component
    - Implement progress bar with percentage
    - Show status messages
    - Handle different states (active, success, error)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 13.2 Create DatasetPreview component
    - Display entries in formatted table
    - Implement JSON formatting
    - Add text truncation with expand
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 13.3 Write property test for JSON formatting
    - **Property 15: JSON formatting is readable**
    - **Validates: Requirements 5.4**

  - [ ]* 13.4 Write property test for text truncation
    - **Property 16: Long text is truncated**
    - **Validates: Requirements 5.5**

  - [ ]* 13.5 Write unit tests for shared components
    - Test ProgressDisplay rendering
    - Test DatasetPreview rendering
    - Test truncation behavior
    - _Requirements: 3.1, 5.1, 5.4, 5.5_

- [x] 14. Implement generation page
  - [x] 14.1 Create GenerationForm component
    - Implement form with all generation parameters
    - Add form validation
    - Handle model selection dropdown
    - _Requirements: 1.1, 1.5, 10.1_

  - [x] 14.2 Create GeneratePage component
    - Integrate GenerationForm
    - Handle form submission
    - Display progress using ProgressDisplay
    - Show results with DatasetPreview
    - Implement download functionality
    - _Requirements: 1.1, 1.2, 1.3, 3.1, 5.1_

  - [x] 14.3 Add configuration save/load
    - Offer to save configuration after completion
    - Implement load from saved configurations
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 14.4 Write property test for save offer
    - **Property 17: Save option offered after operations**
    - **Validates: Requirements 6.1**

  - [ ]* 14.5 Write property test for configuration persistence
    - **Property 18: Configurations persist in storage**
    - **Validates: Requirements 6.2**

  - [ ]* 14.6 Write property test for configuration round-trip
    - **Property 19: Configuration round-trip preserves values**
    - **Validates: Requirements 6.3**

  - [ ]* 14.7 Write unit tests for generation page
    - Test form submission
    - Test progress display
    - Test download functionality
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 15. Implement augmentation page
  - [x] 15.1 Create file upload component
    - Implement drag-and-drop file upload
    - Show file information after upload
    - Display field list
    - _Requirements: 2.1, 4.1_

  - [x] 15.2 Create AugmentationForm component
    - Implement form with augmentation parameters
    - Add field selection from uploaded file
    - Handle context fields selection
    - _Requirements: 2.2, 2.3_

  - [x] 15.3 Create AugmentPage component
    - Integrate file upload and form
    - Implement preview functionality
    - Handle augmentation submission
    - Display progress and results
    - Implement download functionality
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 5.2_

  - [ ]* 15.4 Write unit tests for augmentation page
    - Test file upload
    - Test preview functionality
    - Test form submission
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 16. Implement configuration management page
  - [x] 16.1 Create ConfigPage component
    - Display list of saved configurations
    - Show configuration details (name, type, timestamp)
    - Implement load configuration
    - Implement delete configuration
    - _Requirements: 6.3, 6.4, 6.5_

  - [ ]* 16.2 Write property test for configuration deletion
    - **Property 20: Configuration deletion removes from storage**
    - **Validates: Requirements 6.5**

  - [ ]* 16.3 Write unit tests for config page
    - Test configuration list display
    - Test load functionality
    - Test delete functionality
    - _Requirements: 6.3, 6.4, 6.5_

- [x] 17. Implement language switching
  - [x] 17.1 Create language selector component
    - Add language dropdown to header
    - Implement language switching
    - Save language preference
    - _Requirements: 8.2, 8.5_

  - [x] 17.2 Add translations for all UI text
    - Translate all static text
    - Translate error messages
    - Translate form labels and placeholders
    - _Requirements: 8.2, 8.3_

  - [ ]* 17.3 Write property test for language detection
    - **Property 26: Browser language detection works**
    - **Validates: Requirements 8.1**

  - [ ]* 17.4 Write property test for language switching
    - **Property 27: Language switching updates UI**
    - **Validates: Requirements 8.2**

  - [ ]* 17.5 Write property test for error localization
    - **Property 28: Error messages are localized**
    - **Validates: Requirements 8.3**

  - [ ]* 17.6 Write property test for language persistence
    - **Property 29: Language preference persists**
    - **Validates: Requirements 8.5**

  - [ ]* 17.7 Write unit tests for language switching
    - Test language selector
    - Test translation updates
    - Test preference saving
    - _Requirements: 8.2, 8.5_

- [x] 18. Implement error handling and user feedback
  - [x] 18.1 Create error display components
    - Implement error message display
    - Add retry functionality
    - Show technical details in expandable section
    - _Requirements: 1.4, 2.5, 3.4_

  - [x] 18.2 Add loading states
    - Show loading indicators during API calls
    - Disable forms during submission
    - _Requirements: 3.1_

  - [x] 18.3 Implement offline detection
    - Detect network connectivity
    - Show offline indicator
    - Queue operations for retry
    - _Requirements: 7.5_

  - [ ]* 18.4 Write unit tests for error handling
    - Test error display
    - Test retry functionality
    - Test offline detection
    - _Requirements: 1.4, 3.4, 7.5_

- [x] 19. Checkpoint - Frontend UI complete
  - Ensure all frontend tests pass
  - Verify all pages render correctly
  - Test user interactions
  - Ask the user if questions arise

- [-] 20. Integration and end-to-end testing
  - [x] 20.1 Set up E2E testing framework
    - Install Cypress or Playwright
    - Configure test environment
    - _Requirements: All_

  - [ ]* 20.2 Write E2E tests for generation flow
    - Test complete generation workflow
    - Test progress updates
    - Test download
    - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3_

  - [ ]* 20.3 Write E2E tests for augmentation flow
    - Test file upload
    - Test preview
    - Test complete augmentation workflow
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3_

  - [ ]* 20.4 Write E2E tests for configuration management
    - Test save configuration
    - Test load configuration
    - Test delete configuration
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 20.5 Write E2E tests for error scenarios
    - Test Ollama unavailable
    - Test invalid file upload
    - Test network errors
    - _Requirements: 1.4, 4.4, 7.5_

- [x] 21. Documentation and deployment preparation
  - [x] 21.1 Write API documentation
    - Generate OpenAPI/Swagger docs
    - Add endpoint descriptions
    - Add example requests/responses
    - _Requirements: 9.5_

  - [x] 21.2 Write user documentation
    - Create user guide for web interface
    - Add screenshots
    - Document common workflows
    - _Requirements: All_

  - [x] 21.3 Create deployment configurations
    - Write Dockerfile for backend
    - Write Dockerfile for frontend
    - Create docker-compose.yml
    - Add deployment instructions
    - _Requirements: 9.1_

  - [x] 21.4 Set up production configuration
    - Configure environment variables
    - Set up logging
    - Configure CORS for production
    - _Requirements: 9.1, 9.3_

- [x] 22. Final checkpoint - Complete system testing
  - Run all tests (unit, integration, E2E, property-based)
  - Verify all requirements are met
  - Test deployment with Docker
  - Perform security review
  - Ask the user if questions arise
