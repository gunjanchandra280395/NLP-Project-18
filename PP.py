from tkinter import * 
# from NLP_FINAL import XYZ

insertSentencesWindow = Tk()
insertSentencesWindow.title("Welcome to project number 18")
insertSentencesWindow.geometry('450x400')

# State
numberOfEnteredSentences = 0
S1 = []
S2 = []

# Insert sentences layout string variables
instructionsLabelStringVariable = StringVar()
instructionsLabelStringVariable.set("Enter first sentence")
#resultLabelStringVariable = StringVar()
sentenceInputEntryStringVariable = StringVar() # This variable will contain what ever the user has typed into the input field
#result = StringVar()

# Public functions
def getSentences():
	global S1, S2
	return [S1, S2]

# Callbacks
def submitCallback(): # This function will get called when user presses the 
	global numberOfEnteredSentences, S1, S2
	
	numberOfEnteredSentences = numberOfEnteredSentences + 1

	enteredSentence = sentenceInputEntryStringVariable.get()

	if numberOfEnteredSentences == 1:
		S1 = enteredSentence
		#submitButton = Button(insertSentencesWindow, text= "Process")
	elif numberOfEnteredSentences == 2:
		S2 = enteredSentence

	sentenceInputEntryStringVariable.set("") # Clear the input field
	instructionsLabelStringVariable.set("Enter second sentence") # Update instructions
	
	if numberOfEnteredSentences == 2:
		# Line below is mock functionality: we should query the api here and then update the result resultLabelStringVariable
		#result.set("test: ")
		#resultLabelStringVariable.set("blaablaa")
		insertSentencesWindow.destroy()
		# removeInsertSentencesLayout()
		# displayResultsLayout()

# Insert sentences layout elements
instructionLabel = Label(insertSentencesWindow, textvariable=instructionsLabelStringVariable, font=("Arial Bold", 20))
sentenceInputEntry = Entry(insertSentencesWindow, textvariable=sentenceInputEntryStringVariable)
#resultLabel2 = Label(insertSentencesWindow, textvariable=result)
#resultLabel = Label(insertSentencesWindow, textvariable=resultLabelStringVariable)
submitButton = Button(insertSentencesWindow, text= "submit", width='10', height = '20', command=submitCallback, fg='red')


def displayInsertSentencesLayout():
	instructionLabel.grid(column=3, row=0)
	#instructionLabel.pack()
	
	sentenceInputEntry.grid(column=3, row=1)
	sentenceInputEntry.focus()
	#sentenceInputEntry.pack()
	
	#resultLabel2.pack()
	#resultLabel2.grid(column=2, row=0)


	#resultLabel.grid(column=3, row=0)
	#resultLabel.pack()
	
	submitButton.grid(column=3, row=3)
	#submitButton.pack()
	insertSentencesWindow.mainloop()


# def removeInsertSentencesLayout():
# 	instructionLabel.grid_forget()
# 	sentenceInputEntry.grid_forget()
# 	#resultLabel2.grid_forget()
# 	#resultLabel.grid_forget()
# 	submitButton.grid_forget()


def displayResultsLayout(firstSentence, secondSentence, jaccardResult, ExjaccardResult, wordNetResult, wikiSimilarity):
	# Update labels
	# resultsFirstSentenceStringVariable.set(S1)
	# resultsSecondSentenceStringVariable.set(S2)
	# set variables here############

	# Setup window
	resultsWindow = Tk()
	resultsWindow.title("Welcome to project number 18")
	resultsWindow.geometry('1000x1000')

	# Layout elements
	Label(resultsWindow, text="Sentence 1", font=("Arial Bold", 20)).grid(row=0, column=0)
	Label(resultsWindow, text="Sentence 2", font=("Arial Bold", 20)).grid(row=0, column=1)

	Label(resultsWindow, text=firstSentence).grid(row=1, column=0)
	Label(resultsWindow, text=secondSentence).grid(row=1, column=1)

	# Label(resultsWindow, text="Snippets", font=("Arial Bold", 20)).grid(row=2, column=0)
	# Label(resultsWindow, text=firstSnippet).grid(row=3, column=0)
	# Label(resultsWindow, text=secondSnippet).grid(row=3, column=1)

	Label(resultsWindow, text="Jaccard similarity", font=("Arial Bold", 20)).grid(row=2, column=0)
	Label(resultsWindow, text=jaccardResult).grid(row=3, column=0)

	Label(resultsWindow, text="Expand Jaccard similarity", font=("Arial Bold", 20)).grid(row=4, column=0)
	Label(resultsWindow, text=ExjaccardResult).grid(row=5, column=0)

	Label(resultsWindow, text="WordNet similarity", font=("Arial Bold", 20)).grid(row=6, column=0)
	Label(resultsWindow, text=wordNetResult).grid(row=7, column=0)

	Label(resultsWindow, text="Wiki similarity", font=("Arial Bold", 20)).grid(row=8, column=0)
	Label(resultsWindow, text=wikiSimilarity).grid(row=9, column=0)

	resultsWindow.mainloop()
