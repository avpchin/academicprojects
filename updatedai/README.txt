Usage:
	python game.py <seed> <player class>...

where <seed> may be an integer, which gives a random seed,
or "-", in which case the random default random seed is used.

Up to six player classes can be specified on the command line,
each of which may be one of the following:

- AttentionDeficitPlayer
- PatientPlayer
- GreedyPlayer
- ColorVisionPlayer
- RoutePlayer
- FaceUpPlayer
- LearningPlayer (note: modifies learning.txt)
- LearnedPlayer
- MonteCarloPlayer
- ShoddyUIPlayer (note: requires user input)
- DummyPlayer (note: not a real player, but causes the game to pause each turn)

Examples:

	python game.py - FaceUpPlayer FaceUpPlayer FaceUpPlayer FaceUpPlayer
	python game.py - FaceUpPlayer FaceUpPlayer AttentionDeficitPlayer
	python game.py - LearnedPlayer FaceUpPlayer ShoddyUIPlayer
	python game.py - LearnedPlayer LearnedPlayer
	python game.py 1912 LearnedPlayer AttentionDeficitPlayer FaceUpPlayer
	python game.py 2 LearnedPlayer FaceUpPlayer

We include two files "learning.txt" and "learning.good.txt" which are currently
identical, but learning.txt is modified when the game is played by a LearningPlayer.
learning.txt may be deleted to retrain LearningPlayer from zero weights.

The weights in learning.good.txt were learned in 30740 episodes over the course
of about four days.

Note that LearningPlayer, LearnedPlayer, and MonteCarloPlayer are much slower than
the other players, so a game containing multiple may take a while.

To generate a visualization of the board:

         fdp -T pdf -o board.pdf board.gv

The visualization will then be in board.pdf, which can be opened normally. game.py
automatically updates board.gv after each player's move.
ShoddyUIPlayer and DummyPlayer automatically regenerate board.pdf before each move.
