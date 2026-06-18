# ALON assistant project Objectives

- Speech like LLM (generating response that is suitable for text-to-speech, that means no long response, persice and concrete)
- Some women's voice that looks realistic, emotional (that means no robotic voice)
- Be able to perform search on internet and process the most relavent information online
- Saying weather and saying what to wear that day (if not specified the location, by default take Erevan)
- Listening to a keyword "Alon", when that word said perform voice recognition to be sure that the speaker is me and listening to the end performing speech-to-text to provide the prompt to the LLM
- Saving context in some default file and being able to know the time in Erevan and writing time at the front of the context like day month year and time when the prompt was asked and what she responded
- Having access to my computer information and processes
    - Be able to navigate through directories (that means access to shell)
    - Be able to execute codex by providing a prompt to perform the task specified

Classifier - [Assist], [Weather], [Search], [NavigateAndAssist], [ExecuteCodex]

Pipeline: (Listening for keyword "Alon")->(Voice Recognition)->(Speech-To-Text)->(Classifier of a prompt)->(Taking actions regarding the class of a prompt, generating response)->(Response-to-speech)->(Saving the context with a time)->(Listening for keyword "Alon")
