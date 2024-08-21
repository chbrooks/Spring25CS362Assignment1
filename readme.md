### Introduction to AI
#### Assignment 1 : Getting started

##### Due: Friday Sept 4, 11:59pm.

##### 100 points.

Note: please make sure that your name is someplace in your assignment! This is especially true if it's not obvious from your GitHub handle.

To turn in: please submit everything to your github repo. For the written portion of the assignment, please add a document to your repository, as PDF, containing the answers to these questions.


##### Question 1: Environments. (10 points - 2 points each)
*(milestone Wed Aug 28)*

a) What does it mean for an environment to be stochastic? The self-driving car is an example of a stochastic environment; why is this?

b) What does it mean for an environment to be partially observable? The self-driving car is also partially observable. Why is this, and how does it differ from a stochastic environment?

c) What does it mean for an environment to be sequential? Chess is an example of a sequential environment. Why is this?

d) What does it mean for an environment to be dynamic? A video game is an example of an environment that is usually dynamic. Why is this?

e) What does it mean for an environment to be multiagent? Card games such as poker can be treated as multiagent. How does this approach change the way we think about the problem?

##### Question 2: Approach. (20 points)

*(milestone: Monday Aug 26)* 

A common technique for estimation of complex problems is Monte Carlo simulation. It's also our first exposure to <i> generative</i> algorithms, which are algorithms that
can generate novel data from a distribution. We'll use it here to try to estimate 
an optimal stopping point (or policy) for a simple dice game, called approach. This will also give you some practice with Python.

Approach works like this: 

There are two players, and they choose a target number (n).
Player 1 repeatedly rolls a six-sided die and adds up their total, while trying to stay below or equal to the target. 
Whenever they want, they can "hold." Then player 2 must roll. Their goal is 
to beat (not tie) player 1's score without going past the target. The problem we want to solve is this:

For player 1, for a given target and a particular total so far should they: 
'hold' (action 1) or 'roll' (action 2).

If we think a little about the game, we can see that once we've determined the lowest value at which we should hold, we know what to do for greater values. In other words, if I've decided that I should hold at 6, then I should also hold for 7-10. So the question now becomes, what's the lowest number where we should hold? Let's call this the hold value. So what's the best hold value?

To figure this out, we'll use a brute force approach. We'll try to estimate the probability of winning for each possible hold value. Then we can just choose the hold value with the highest probability.

(note that player 2's strategy in this game is pretty boring - they just roll until they either beat player 1's score or exceed n.)

I've provided the beginnings of this program for you. 
It should: take as input a target value n.:
  For all values from n-5:n, estimate the probability of winning, given that we use that value as our hold value.

For example, if n=10, you might see:
<pre>
monte_carlo_approach(10)
5: 0.255314
6: 0.360087
7: 0.431315
8: 0.435367
9: 0.371685
10: 0.205657
</pre>

Since our probability of winning is highest when hold=8, we should roll when we're at 7 or lower, and hold on 8, 9 or 10.

To estimate these probabilities, we'll play the game 1000000 times for each possible hold. 
A round of the game works like this:
   - player 1 sets their hold value. They then 'roll the die' (generate a random int between 1 and 6) and add to their score until:
     - They exceed n, in which case they lose.
     - Their total is >= to their hold value, and less than or equal to n. In this case player 2 goes.
       - Player 2 rolls until their total either:
         - Exceeds player 1, but is less than or equal to n. Player 1 loses.
         - Exceeds n. Player 1 wins.

Please note that there are a few edge cases in here. (What if they tie? What if they both tie at the target?) You may choose 
how to implement this; just be consistent in your results.
         
 
##### Question 3. (20 points) Machine Learning preparation
*(milestone Aug 30)*

In this question, we'll start building the tools and knowledge that we'll need to do *machine learning.* 
For this assignment, we'll start with a toy dataset; the tennis dataset. You'll see this a few more times throughout the semester. It's small and boring, but easy to work with.

We're going to start by implementing two baseline techniques: ZeroR and RandR. These will be our classifiers of last resort later on; when
we don't know what else to do, we'll use them.

They're both very simple. ZeroR just looks at the data and returns the most common classification. So, if 60% of our data says "yes", ZeroR will always return 'yes'.

RandR looks at the data and randomly selects a classification from the data according to their frequency. So if 60% of the data is 'yes', it will return 'yes'
with probability 0.6 and 'no' with p=0.4.

Why would we want to do this? Two reasons: 1) sometimes the classification is all we have, and the features are not helpful. 2) 
It's a useful benchmark for lower bounds. If our new learning algorithm cannot perform better than these, then we should probably discard it.

I've provided a skeleton; you fill in the ZeroR and RandR functions and then run them on the tennis dataset to compute <i>accuracy</i>, which is the fraction of instances in the dataset that are correctly classified.

##### Question 4 (20 points): Word frequencies. 
*(milestone: Sept 1)*

One of the primary domains we'll  work with this semester is text. The most common approach to dealing with
  large bodies of text is statistical. This involves converting a document into a *vector*. This requires counting the number of words in a document.

wc.py gives you a starting point for a program that will read words from a file and construct a frequency distribution. In this question you'll extend wc.py; this will also give you the opportunity to explore the Python standard library. 

Please add the following features - 
you may need to explore the Python documentation a little to figure these out.

1. The way we're processing command-line args is not very extensible. Also, we're not using standard tools.
Please replace this with the [argparse module](https://docs.python.org/3/library/argparse.html).

2. The current wc.py will only process a single file. Fix it to be able to take as input a directory name or path and call wc for all files in that directory, or subdirectories. (look at [os.walk()](https://docs.python.org/3/library/os.html) to see how to do this.) Please note: **all** files means everything 
that is text, regardless of extension. You can skip over non-text files. I would suggest doing this with a try/except block.

3. We will find it helpful to be able to store this dictionary so that we can use it later on. If the --save=file option is present, write the dictionary to a file **as an object** using the pickle module*.

4. We have options for stripping punctuation and lowercasing. We would like to add the following options:
   -  Remove non-words. For this assignment, we'll assume that any string which contains a nonalphabetic character is a nonword. So cat123, !rabbit, and one-sized are all nonwords. Add a command-line option --nonwords and extend the wordfreq function to account for this option.
   - Change the separator. Right now we are assuming that space is the best character to split on. Add a command-line argument --separator=string. You should use **any** character within string as a separator. So if the user provides --separator="#;." you would use the #, ;, or . as a separator. (Hint: get this working with single characters first, then 
look at the re module to see how to use a regular expression to do this.)
   

#### Question 5. (20 points) Ollama
*(milestone: sept 3)*

In this question, you'll get some initial experience working with a Large Language Model. We'll be using [ollama](https://ollama.com/), which provides both a command-line
interface and a Python API for your favorite LLM. 

- To begin, get Ollama installed on your computer. The GitHub repo is [here](https://github.com/ollama/ollama). 
There are lots of ways to do it - you can download the installer, or use a Docker container if you prefer. The goal is to be ale to do this:
(If your computer doesn't have enough RAM, ssh into one of the lab computers, or use the CS Virtual Lab. (you should have learned about this in
CS221; if you don't know about it, please ask!)

This is a great example of a place where it's great to work together and help each other. If you're having problems, please post them on Piazza! 

- Next, create a Modelfile that contains your own custom system prompt. (add this to your repo.) 

- Next, install the Ollama Python library (with pip install ollama).

- Last, let's see if we can set up a simple Ollama server. This will listen on a socket for a request, call the LLM, a nd return a reply.
You can find the Ollama-python docs[here](https://github.com/ollama/ollama-python).

Start by creating a server, like so:

    import ollama
    response = ollama.chat(model='llama3.1', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
    ])
    print(response['message']['content'])

You can then call it from the command line like this:

    curl http://localhost:11434/api/generate -d '{
    "model": "llama3",
    "prompt": "Why is the sky blue?"
    }'

You should see a whole bunch of responses, one for each token generated, like so:

    {"model":"llama3","created_at":"2024-08-20T21:20:24.919781Z","response":"The","done":false}
    {"model":"llama3","created_at":"2024-08-20T21:20:24.952087Z","response":" sky","done":false}
    {"model":"llama3","created_at":"2024-08-20T21:20:24.984478Z","response":" appears","done":false}

Please include a Python file with the code for your Ollama server in your repo.

Congratulations! You've learned how to deal with Python packages, gotten a server installed, and learned a little
about LLM configuration! We'll delve more into this later in the semester.

##### Question 6 (10 points): Detecting Intelligence. 

*(milestone: Sept 4)*

Large Language Models, or LLMs, have had a massive impact both on the field of AI and on culture more broadly since their introduction a few years ago.

For this question, we'll consider three of the most well-known LLMS:

- [ChatGPT](https://chat.openai.com/auth/login) from OpenAI.</li>
- [Gemini](https://gemini.google.com/) from Google </li>
- [Claude](https://claude.ai/login) from Anthropic. </li>

You will need to create accounts for each of these, but should not need to sign up for any paid services.

Choose one of the environment questions from question 1, and paste it into one of  these LLMs, and provide a copy of their response.
Are they correct? Can you distinguish between your answer and an LLM?

Now pick one of the LLMs and craft a prompt in which you give it the original question, your answer, and an answer from one of the other LLMs, and ask it to determine which answer was written by a human and why. Does it get the answer right?

 Include a copy of your prompt and the LLM's reply. 

##### Question 7: (Grad Students only):  

A classic thought experiment in response to the Turing Test is the Chinese Room, proposed by John Searle.
To begin, please read [this webpage](https://mind.ilstu.edu/curriculum/searle_chinese_room/searle_chinese_room.html), which presents the Chinese Room thought experiment, intended to show that a computer can manipulate symbols to produce replies without understanding what they mean.

Then, read section 2 (Replies and Rejoinders) of [this page](https://iep.utm.edu/chineser/), which summarizes many of the responses to the Chinese Room argument.

a) Summarize the Chinese Room argument.

b) What does this have to do with computers? 

c) Why does Searle believe that it shows that a computer can pass the Turing test without understanding?

d) Do you find Searle's argument convincing? Or do you find one of the responses more appealing? Explain your position. 