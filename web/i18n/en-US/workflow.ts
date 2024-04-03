const translation = {
  common: {
    editing: 'Editing',
    autoSaved: 'Auto-Saved',
    unpublished: 'Unpublished',
    published: 'Published',
    publish: 'Publish',
    update: 'Update',
    run: 'Run',
    running: 'Running',
    inRunMode: 'In Run Mode',
    inPreview: 'In Preview',
    inPreviewMode: 'In Preview Mode',
    preview: 'Preview',
    viewRunHistory: 'View run history',
    runHistory: 'Run History',
    goBackToEdit: 'Go back to editor',
    conversationLog: 'Conversation Log',
    features: 'Features',
    debugAndPreview: 'Debug and Preview',
    restart: 'Restart',
    currentDraft: 'Current Draft',
    currentDraftUnpublished: 'Current Draft Unpublished',
    latestPublished: 'Latest Published',
    publishedAt: 'Published',
    restore: 'Restore',
    runApp: 'Run App',
    batchRunApp: 'Batch Run App',
    accessAPIReference: 'Access API Reference',
    embedIntoSite: 'Embed Into Site',
    addTitle: 'Add title...',
    addDescription: 'Add description...',
    noVar: 'No variable',
    searchVar: 'Search variable',
    variableNamePlaceholder: 'Variable name',
    setVarValuePlaceholder: 'Set variable',
    needConnecttip: 'This step is not connected to anything',
    maxTreeDepth: 'Maximum limit of {{depth}} nodes per branch',
    needEndNode: 'Please add an end node',
    needAnswerNode: 'Please add a answer node',
    workflowProcess: 'Workflow Process',
    notRunning: 'Not running yet',
    previewPlaceholder: 'Enter content in the box below to start debugging the Chatbot',
    effectVarConfirm: {
      title: 'Remove Variable',
      content: 'The variable is used in other nodes. Do you still want to remove it?',
    },
    insertVarTip: 'Press the \'/\' key to insert quickly',
  },
  errorMsg: {
    fieldRequired: '{{field}} is required',
    invalidJson: '{{field}} is invalid JSON',
    fields: {
      variable: 'Variable Name',
      variableValue: 'Variable Value',
      code: 'Code',
      model: 'Model',
      rerankModel: 'Rerank Model',
    },
  },
  singleRun: {
    testRun: 'Test Run ',
    startRun: 'Start Run',
    running: 'Running',
  },
  tabs: {
    'searchBlock': 'Search block',
    'blocks': 'Blocks',
    'builtInTool': 'Built-in Tool',
    'customTool': 'Custom Tool',
    'question-understand': 'Question Understand',
    'logic': 'Logic',
    'transform': 'Transform',
    'utilities': 'Utilities',
    'noResult': 'No match found',
  },
  blocks: {
    'start': 'Start',
    'end': 'End',
    'answer': 'Answer',
    'llm': 'LLM',
    'knowledge-retrieval': 'Knowledge Retrieval',
    'question-classifier': 'Question Classifier',
    'if-else': 'IF/ELSE',
    'code': 'Code',
    'template-transform': 'Template',
    'http-request': 'HTTP Request',
    'variable-assigner': 'Variable Assigner',
  },
  blocksAbout: {
    'start': 'Define the initial parameters for launching a workflow',
    'end': 'Define the end and result type of a workflow',
    'answer': 'Define the reply content of a chat conversation',
    'llm': 'Invoking large language models to answer questions or process natural language',
    'knowledge-retrieval': 'Allows you to query text content related to user questions from the Knowledge',
    'question-classifier': 'Define the classification conditions of user questions, LLM can define how the conversation progresses based on the classification description',
    'if-else': 'Allows you to split the workflow into two branches based on if/else conditions',
    'code': 'Execute a piece of Python or NodeJS code to implement custom logic',
    'template-transform': 'Convert data to string using Jinja template syntax',
    'http-request': 'Allow server requests to be sent over the HTTP protocol',
    'variable-assigner': 'Assign variables in different branches to the same variable to achieve unified configuration of post-nodes',
  },
  operator: {
    zoomIn: 'Zoom In',
    zoomOut: 'Zoom Out',
    zoomTo50: 'Zoom to 50%',
    zoomTo100: 'Zoom to 100%',
    zoomToFit: 'Zoom to Fit',
  },
  panel: {
    userInputField: 'User Input Field',
    changeBlock: 'Change Block',
    helpLink: 'Help Link',
    about: 'About',
    createdBy: 'Created By ',
    nextStep: 'Next Step',
    addNextStep: 'Add the next block in this workflow',
    selectNextStep: 'Select Next Block',
    runThisStep: 'Run this step',
    checklist: 'Checklist',
    checklistTip: 'Make sure all issues are resolved before publishing',
    checklistResolved: 'All issues are resolved',
    organizeBlocks: 'Organize blocks',
    change: 'Change',
  },
  nodes: {
    common: {
      outputVars: 'Output Variables',
      insertVarTip: 'Insert Variable',
      memory: {
        memory: 'Memory',
        memoryTip: 'Chat memory settings',
        windowSize: 'Window Size',
        conversationRoleName: 'Conversation Role Name',
        user: 'User prefix',
        assistant: 'Assistant prefix',
      },
      memories: {
        title: 'Memories',
        tip: 'Chat memory',
        builtIn: 'Built-in',
      },
    },
    start: {
      required: 'required',
      inputField: 'Input Field',
      builtInVar: 'Built-in Variables',
      outputVars: {
        query: 'User input',
        memories: {
          des: 'Conversation history',
          type: 'message type',
          content: 'message content',
        },
        files: 'File list',
      },
      noVarTip: 'Set inputs that can be used in the Workflow',
    },
    end: {
      outputs: 'Outputs',
      output: {
        type: 'output type',
        variable: 'output variable',
      },
      type: {
        'none': 'None',
        'plain-text': 'Plain Text',
        'structured': 'Structured',
      },
    },
    answer: {
      answer: 'Answer',
      outputVars: 'Output Variables',
    },
    llm: {
      model: 'model',
      variables: 'variables',
      context: 'context',
      contextTooltip: 'You can import Knowledge as context',
      prompt: 'prompt',
      roleDescription: {
        system: 'Give high level instructions for the conversation',
        user: 'Provide instructions, queries, or any text-based input to the model',
        assistant: 'The model’s responses  based on the user messages',
      },
      addMessage: 'Add Message',
      vision: 'vision',
      files: 'Files',
      resolution: {
        name: 'Resolution',
        high: 'High',
        low: 'Low',
      },
      outputVars: {
        output: 'Generate content',
        usage: 'Model Usage Information',
      },
      singleRun: {
        variable: 'Variable',
      },
    },
    knowledgeRetrieval: {
      queryVariable: 'Query Variable',
      knowledge: 'Knowledge',
      outputVars: {
        output: 'Retrieval segmented data',
        content: 'Segmented content',
        title: 'Segmented title',
        icon: 'Segmented icon',
        url: 'Segmented URL',
        metadata: 'Other metadata',
      },
    },
    http: {
      inputVars: 'Input Variables',
      api: 'API',
      apiPlaceholder: 'Enter URL, type ‘/’ insert variable',
      notStartWithHttp: 'API should start with http:// or https://',
      key: 'Key',
      value: 'Value',
      bulkEdit: 'Bulk Edit',
      keyValueEdit: 'Key-Value Edit',
      headers: 'Headers',
      params: 'Params',
      body: 'Body',
      outputVars: {
        body: 'Response Content',
        statusCode: 'Response Status Code',
        headers: 'Response Header List JSON',
        files: 'Files List',
      },
      authorization: {
        'authorization': 'Authorization',
        'authorizationType': 'Authorization Type',
        'no-auth': 'None',
        'api-key': 'API-Key',
        'auth-type': 'Auth Type',
        'basic': 'Basic',
        'bearer': 'Bearer',
        'custom': 'Custom',
        'api-key-title': 'API Key',
        'header': 'Header',
      },
      insertVarPlaceholder: 'type \'/\' to insert variable',
    },
    code: {
      inputVars: 'Input Variables',
      outputVars: 'Output Variables',
    },
    templateTransform: {
      inputVars: 'Input Variables',
      code: 'Code',
      codeSupportTip: 'Only supports Jinja2',
      outputVars: {
        output: 'Transformed content',
      },
    },
    ifElse: {
      if: 'If',
      else: 'Else',
      elseDescription: 'Used to define the logic that should be executed when the if condition is not met.',
      and: 'and',
      or: 'or',
      operator: 'Operator',
      notSetVariable: 'Please set variable first',
      comparisonOperator: {
        'contains': 'contains',
        'not contains': 'not contains',
        'start with': 'start with',
        'end with': 'end with',
        'is': 'is',
        'is not': 'is not',
        'empty': 'is empty',
        'not empty': 'is not empty',
        'null': 'is null',
        'not null': 'is not null',
      },
      enterValue: 'Enter value',
      addCondition: 'Add Condition',
      conditionNotSetup: 'Condition NOT setup',
    },
    variableAssigner: {
      title: 'Assign variables',
      outputType: 'Output Type',
      outputVarType: 'Output Variable Type',
      varNotSet: 'Variable not set',
      noVarTip: 'Add the variables to be assigned',
      type: {
        string: 'String',
        number: 'Number',
        object: 'Object',
        array: 'Array',
      },
      outputVars: {
        output: 'Assigned variable value',
      },
    },
    tool: {
      toAuthorize: 'To authorize',
      inputVars: 'Input Variables',
      outputVars: {
        text: 'tool generated content',
        files: {
          title: 'tool generated files',
          type: 'Support type. Now only support image',
          transfer_method: 'Transfer method.Value is  remote_url or local_file',
          url: 'Image url',
          upload_file_id: 'Upload file id',
        },
      },
    },
    questionClassifiers: {
      model: 'model',
      inputVars: 'Input Variables',
      class: 'Class',
      classNamePlaceholder: 'Write your class name',
      advancedSetting: 'Advanced Setting',
      topicName: 'Topic Name',
      topicPlaceholder: 'Write your topic name',
      addClass: 'Add Class',
      instruction: 'Instruction',
      instructionPlaceholder: 'Write your instruction',
    },
  },
  tracing: {
    stopBy: 'Stop by {{user}}',
  },
}

export default translation