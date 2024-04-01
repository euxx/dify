import { useCallback, useEffect, useRef, useState } from 'react'
import produce from 'immer'
import { flatten, uniqBy } from 'lodash'
import useVarList from '../_base/hooks/use-var-list'
import { VarType } from '../../types'
import type { Memory, ValueSelector, Var } from '../../types'
import { useStore } from '../../store'
import {
  useIsChatMode,
  useNodesReadOnly,
  useWorkflow,
} from '../../hooks'
import { getNodeInfoById } from '../_base/components/variable/utils'
import type { LLMNodeType } from './types'
import { Resolution } from '@/types/app'
import { useModelListAndDefaultModelAndCurrentProviderAndModel, useTextGenerationCurrentProviderAndModelAndModelList } from '@/app/components/header/account-setting/model-provider-page/hooks'
import { ModelFeatureEnum } from '@/app/components/header/account-setting/model-provider-page/declarations'
import useNodeCrud from '@/app/components/workflow/nodes/_base/hooks/use-node-crud'
import useOneStepRun from '@/app/components/workflow/nodes/_base/hooks/use-one-step-run'
import type { PromptItem } from '@/models/debug'
import { RETRIEVAL_OUTPUT_STRUCT } from '@/app/components/workflow/constants'
import { checkHasContextBlock, checkHasHistoryBlock, checkHasQueryBlock, getInputVars } from '@/app/components/base/prompt-editor/constants'

const useConfig = (id: string, payload: LLMNodeType) => {
  const { nodesReadOnly: readOnly } = useNodesReadOnly()
  const isChatMode = useIsChatMode()
  const { getBeforeNodesInSameBranch } = useWorkflow()

  const availableNodes = getBeforeNodesInSameBranch(id)

  const defaultConfig = useStore(s => s.nodesDefaultConfigs)[payload.type]
  const [defaultRolePrefix, setDefaultRolePrefix] = useState<{ user: string; assistant: string }>({ user: '', assistant: '' })
  const { inputs, setInputs: doSetInputs } = useNodeCrud<LLMNodeType>(id, payload)
  const setInputs = useCallback((newInputs: LLMNodeType) => {
    if (newInputs.memory && !newInputs.memory.role_prefix) {
      const newPayload = produce(newInputs, (draft) => {
        draft.memory!.role_prefix = defaultRolePrefix
      })
      doSetInputs(newPayload)
      return
    }
    doSetInputs(newInputs)
  }, [doSetInputs, defaultRolePrefix])
  const inputRef = useRef(inputs)
  useEffect(() => {
    inputRef.current = inputs
  }, [inputs])
  // model
  const model = inputs.model
  const modelMode = inputs.model?.mode
  const isChatModel = modelMode === 'chat'

  const isCompletionModel = !isChatModel

  const hasSetBlockStatus = (() => {
    const promptTemplate = inputs.prompt_template
    const hasSetContext = isChatModel ? (promptTemplate as PromptItem[]).some(item => checkHasContextBlock(item.text)) : checkHasContextBlock((promptTemplate as PromptItem).text)
    if (!isChatMode) {
      return {
        history: false,
        query: false,
        context: hasSetContext,
      }
    }
    if (isChatModel) {
      return {
        history: false,
        query: (promptTemplate as PromptItem[]).some(item => checkHasQueryBlock(item.text)),
        context: hasSetContext,
      }
    }
    else {
      return {
        history: checkHasHistoryBlock((promptTemplate as PromptItem).text),
        query: checkHasQueryBlock((promptTemplate as PromptItem).text),
        context: hasSetContext,
      }
    }
  })()

  const appendDefaultPromptConfig = useCallback((draft: LLMNodeType, defaultConfig: any, passInIsChatMode?: boolean) => {
    const promptTemplates = defaultConfig.prompt_templates
    if (passInIsChatMode === undefined ? isChatModel : passInIsChatMode) {
      draft.prompt_template = promptTemplates.chat_model.prompts
    }
    else {
      draft.prompt_template = promptTemplates.completion_model.prompt

      setDefaultRolePrefix({
        user: promptTemplates.completion_model.conversation_histories_role.user_prefix,
        assistant: promptTemplates.completion_model.conversation_histories_role.assistant_prefix,
      })
    }
  }, [isChatModel])
  useEffect(() => {
    const isReady = defaultConfig && Object.keys(defaultConfig).length > 0

    if (isReady && !inputs.prompt_template) {
      const newInputs = produce(inputs, (draft) => {
        appendDefaultPromptConfig(draft, defaultConfig)
      })
      setInputs(newInputs)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultConfig, isChatModel])

  const {
    currentProvider,
    currentModel,
  } = useModelListAndDefaultModelAndCurrentProviderAndModel(1)

  const handleModelChanged = useCallback((model: { provider: string; modelId: string; mode?: string }) => {
    const newInputs = produce(inputRef.current, (draft) => {
      draft.model.provider = model.provider
      draft.model.name = model.modelId
      draft.model.mode = model.mode!
      const isModeChange = model.mode !== inputRef.current.model.mode
      if (isModeChange && defaultConfig && Object.keys(defaultConfig).length > 0)
        appendDefaultPromptConfig(draft, defaultConfig, model.mode === 'chat')
    })
    setInputs(newInputs)
  }, [setInputs, defaultConfig, appendDefaultPromptConfig])

  useEffect(() => {
    if (currentProvider?.provider && currentModel?.model && !model.provider) {
      handleModelChanged({
        provider: currentProvider?.provider,
        modelId: currentModel?.model,
        mode: currentModel?.model_properties?.mode as string,
      })
    }
  }, [model.provider, currentProvider, currentModel, handleModelChanged])

  const handleCompletionParamsChange = useCallback((newParams: Record<string, any>) => {
    const newInputs = produce(inputs, (draft) => {
      draft.model.completion_params = newParams
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const {
    currentModel: currModel,
  } = useTextGenerationCurrentProviderAndModelAndModelList(
    {
      provider: model.provider,
      model: model.name,
    },
  )
  const isShowVisionConfig = !!currModel?.features?.includes(ModelFeatureEnum.vision)

  // variables
  const { handleVarListChange, handleAddVariable } = useVarList<LLMNodeType>({
    inputs,
    setInputs,
  })

  // context
  const handleContextVarChange = useCallback((newVar: ValueSelector | string) => {
    const newInputs = produce(inputs, (draft) => {
      draft.context.variable_selector = newVar as ValueSelector || []
      draft.context.enabled = !!(newVar && newVar.length > 0)
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handlePromptChange = useCallback((newPrompt: PromptItem[] | PromptItem) => {
    const newInputs = produce(inputs, (draft) => {
      draft.prompt_template = newPrompt
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleMemoryChange = useCallback((newMemory?: Memory) => {
    const newInputs = produce(inputs, (draft) => {
      draft.memory = newMemory
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const handleVisionResolutionChange = useCallback((newResolution: Resolution) => {
    const newInputs = produce(inputs, (draft) => {
      if (!draft.vision.configs) {
        draft.vision.configs = {
          detail: Resolution.high,
        }
      }
      draft.vision.configs.detail = newResolution
    })
    setInputs(newInputs)
  }, [inputs, setInputs])

  const filterInputVar = useCallback((varPayload: Var) => {
    return [VarType.number, VarType.string].includes(varPayload.type)
  }, [])

  const filterVar = useCallback((varPayload: Var) => {
    return [VarType.arrayObject, VarType.array, VarType.string].includes(varPayload.type)
  }, [])

  // single run
  const {
    isShowSingleRun,
    hideSingleRun,
    toVarInputs,
    runningStatus,
    handleRun,
    handleStop,
    runInputData,
    setRunInputData,
    runResult,
  } = useOneStepRun<LLMNodeType>({
    id,
    data: inputs,
    defaultRunInputData: {
      '#context#': [RETRIEVAL_OUTPUT_STRUCT],
      '#files#': [],
    },
  })

  // const handleRun = (submitData: Record<string, any>) => {
  //   console.log(submitData)
  //   const res = produce(submitData, (draft) => {
  //     debugger
  //     if (draft.contexts) {
  //       draft['#context#'] = draft.contexts
  //       delete draft.contexts
  //     }
  //     if (draft.visionFiles) {
  //       draft['#files#'] = draft.visionFiles
  //       delete draft.visionFiles
  //     }
  //   })

  //   doHandleRun(res)
  // }

  const inputVarValues = (() => {
    const vars: Record<string, any> = {}
    Object.keys(runInputData)
      .filter(key => !['#context#', '#files#'].includes(key))
      .forEach((key) => {
        vars[key] = runInputData[key]
      })
    return vars
  })()

  const setInputVarValues = useCallback((newPayload: Record<string, any>) => {
    const newVars = {
      ...newPayload,
      '#context#': runInputData['#context#'],
      '#files#': runInputData['#files#'],
    }
    setRunInputData(newVars)
  }, [runInputData, setRunInputData])

  const contexts = runInputData['#context#']
  const setContexts = useCallback((newContexts: string[]) => {
    setRunInputData({
      ...runInputData,
      '#context#': newContexts,
    })
  }, [runInputData, setRunInputData])

  const visionFiles = runInputData['#files#']
  const setVisionFiles = useCallback((newFiles: any[]) => {
    setRunInputData({
      ...runInputData,
      '#files#': newFiles,
    })
  }, [runInputData, setRunInputData])
  const variables = (() => {
    let valueSelectors: ValueSelector[] = []
    if (isChatModel) {
      valueSelectors = flatten(
        (inputs.prompt_template as PromptItem[])
          .map(item => getInputVars(item.text)),
      )
    }
    else {
      valueSelectors = getInputVars((inputs.prompt_template as PromptItem).text)
    }

    const variables = uniqBy(valueSelectors, item => item.join('.')).map((item) => {
      const varInfo = getNodeInfoById(availableNodes, item[0])?.data
      const variable = [...item]
      variable[0] = varInfo?.title || availableNodes[0]?.data.title // default start node title

      return {
        variable: variable.join('/'),
        value_selector: item,
      }
    })

    return variables
  })()
  const varInputs = toVarInputs(variables)

  return {
    readOnly,
    isChatMode,
    inputs,
    isChatModel,
    isCompletionModel,
    hasSetBlockStatus,
    isShowVisionConfig,
    handleModelChanged,
    handleCompletionParamsChange,
    handleVarListChange,
    handleAddVariable,
    handleContextVarChange,
    filterInputVar,
    filterVar,
    handlePromptChange,
    handleMemoryChange,
    handleVisionResolutionChange,
    isShowSingleRun,
    hideSingleRun,
    inputVarValues,
    setInputVarValues,
    visionFiles,
    setVisionFiles,
    contexts,
    setContexts,
    varInputs,
    runningStatus,
    handleRun,
    handleStop,
    runResult,
  }
}

export default useConfig