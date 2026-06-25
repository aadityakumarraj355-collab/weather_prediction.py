#hangman game
import random

words = ["apple", "mango", "kitten", "penguin", "elephant"]

word = random.choice(words)
guessed = ["_"] * len(word)

attempts = 6
guessed_letters = []

print("Welcome to a simple hangman game")
print("Guess the word")
print(" ".join(guessed))

while attempts > 0 and "_" in guessed:
    guess = input("\nEnter a letter: ").lower()
    
    if not guess.isalpha() or len(guess) != 1:
        print("Please enter only alphabet letters: ")
        continue
        
    if guess in guessed_letters:
        print("\nYou've already guess that letter")
        continue
        
    guessed_letters.append(guess)
    
    if guess in word:
        print("Correct")
        
        for i in range(len(word)):
            if word[i] == guess:
                guessed[i] = guess
    else:
        attempts -= 1
        print("Wrong! Attempts left: ", attempts)
        
    print(" ".join(guessed))

if "_" not in guessed:
    print("Congragulations! You win!")
    print("The word was", word)
else:
    print("Better luck next time")
    print("The word was", word)