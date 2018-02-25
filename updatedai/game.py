# Julia Goldman, Michael Zuo, Kar Yern Chin
# Ticket to Ride
import random
import subprocess
import Queue
import copy
import util
import math

class Game:
    """
    A class holding game state.
    """
    scoring = {1 : 1, 2 : 2, 3 : 4, 4 : 7, 5 : 10, 6 : 15} # #dict mapping costs to points
    colors = ["blue", "red", "orange", "purple", "black", "yellow"]
    initialRoutes = None
    initialDeck = None
    def __init__(self, players = None):
        self.graph = Graph() #game board
        if Game.initialRoutes == None:
            with open("routes.txt") as routes:
                Game.initialRoutes = []
                for line in routes.readlines():
                    city1, city2, dist = line.split(",")
                    Game.initialRoutes.append((city1, city2, int(dist)))
        self.routeDeck = copy.copy(Game.initialRoutes) #list of route tuples of the form (name, name, value)
        self.fakeTrainDeck = []
        random.shuffle(self.routeDeck)
        # counts of cards in deck. We never need the specific order of the
        # cards in the train deck or discard pile, so it's OK for the order to
        # be not determined until we draw a card
        if Game.initialDeck == None:
            with open("colors.txt") as colors:
                Game.initialDeck = {}
                for line in colors.readlines():
                    color, count = line.split()
                    Game.initialDeck[color] = int(count)
        self.trainDeck = copy.copy(Game.initialDeck) #dictionary mapping strings to counts
        # dict with the same keys as the train deck, where cards get "moved"
        # after they've been consumed
        self.discards = {j: 0 for j in self.trainDeck}
        # face-up cards
        self.revealedTrains = [None for _ in range(5)]
        self.revealTrainCards()
        if players != None:
            self.players = [player(self, i) for i, player in enumerate(players)] #list of player objects in turn order

    def insertPlayers(self, players):
        """
        Used by Monte Carlo player to run a simulation without modifying the
        real game state.
        """
        self.players = players

    def drawRouteCards(self):
        """
        Destructively removes and the top three route cards from the route deck and returns them.
        The player must return an appropriate number of them afterward.
        """
        if len(self.routeDeck) >= 3:
            pick = self.routeDeck[:3]
            self.routeDeck = self.routeDeck[3:]
        else:
            pick = self.routeDeck
            self.routeDeck = []
        return pick

    def estimateRouteCompletion(self,pIndex):
        publicInfo = PublicInfo(pIndex, self)
        publicInfo.estimateRouteCompletion()
        self.players[pIndex].score = publicInfo.scores[pIndex]

    def putRouteCards(self, routes):
        """
        Places route cards at the bottom of the deck.
        Must be called by players for any route cards they don't keep,
        so that they are not lost from the game.
        NOTE: must be safe to call multiple or zero times for the same
        instance of drawing route cards, because the human interface
        relies on this.
        """
        for route in routes:
            self.routeDeck.append(route)

    def drawTrainCard(self):
        """
        Destructively remove a card from the train deck and return it.
        """
        total = sum(self.trainDeck.values())
        if total <= 0:
            print "swapping deck with discards"
            self.trainDeck, self.discards = self.discards, self.trainDeck
            total = sum(self.trainDeck.values())
        # This can go wrong if both deck and discards are empty. We manage to
        # avoid this by now allowing our agents to hoard cards.
        # Picks a random key from the dictionary, weighted by value.
        pick = random.randint(1, total)
        for key in self.trainDeck:
            if pick <= self.trainDeck[key]:
                self.trainDeck[key] -= 1
                return key
            pick -= self.trainDeck[key]
        assert False, "somehow we picked a number that's too big"

    def drawFakeTrainCard(self):
        """
        Destructively remove a card from the fake train deck and return it.
        """
        total = len(self.fakeTrainDeck)
        return self.fakeTrainDeck.pop(0)

    def revealTrainCards(self):
        """
        Place train cards face up in the spaces where cards were taken.
        Toss out the revealed cards and draw more if we have too many locomotives.
        """
        for i in range(len(self.revealedTrains)):
            if self.revealedTrains[i] == None:
                self.revealedTrains[i] = self.drawTrainCard()
        # gives up after 10 tries if we can't draw <3 nonrainbow cards
        maxTimes = 10
        while self.revealedTrains.count("rainbow") >= 3 and maxTimes > 0:
            maxTimes -= 1
            print "replacing the things"
            for i in range(len(self.revealedTrains)):
                self.discards[self.revealedTrains[i]] += 1
                self.revealedTrains[i] = self.drawTrainCard()

    def revealFakeTrainCards(self):
        """
        Replaces cards taken from revealed cards with trains from the fake
        train deck used by simulations.
        """
        for i in range(len(self.revealedTrains)):
            if self.revealedTrains[i] == None:
                self.revealedTrains[i] = self.drawFakeTrainCard()
        maxTimes = 10
        while self.revealedTrains.count("rainbow") >= 3 and maxTimes > 0:
            maxTimes -= 1
            for i in range(len(self.revealedTrains)):
                self.discards[self.revealedTrains[i]] += 1
                self.revealedTrains[i] = self.drawFakeTrainCard()

    def checkOver(self):
        """
        Check whether the end condition for the game has been satisfied.
        """
        for player in self.players:
            if player.numTrains < 3:
                return True
        return False

    def sanityCheck(self):
        """
        Make sure state still makes sense at the beginning of the turn
        """
        totalCards = sum(self.trainDeck.values()) + sum(self.discards.values())
        totalCards += sum(1 for card in self.revealedTrains if card)
        for p in self.players:
            totalCards += sum(p.hand.values())
        assert totalCards == 110, "cards got lost"

        totalRoutes = len(self.routeDeck)
        for p in self.players:
            totalRoutes += len(p.routes)
        assert totalRoutes == 30, "routes got lost"

    def playGame(self):
        """
        Main loop.
        """
        turn = 0
        # Initial route card pickup
        for i, player in enumerate(self.players):
            print "player", i, Game.colors[i]
            print ""
            player.pickRouteCards(1)
        # Players take turns until end condition is met
        while not self.checkOver():
            turn += 1
            print "\n\n\nTURN NUMBER " + str(turn)
            for i, player in enumerate(self.players):
                self.sanityCheck()
                if self.checkOver():
                    break
                player.takeAction()

                import os
                #filename = "%d-turn%03d-player%d" % (os.getpid(), turn, i)
                filename = "board"
                visual = open(filename + ".gv", 'w')
                visual.write(str(self.graph))
                visual.close()
                #subprocess.call(["fdp", "-Tpdf", "-o", filename + ".pdf", filename + ".gv"])

        # Game over. Update each player's score with points for
        # complete/incomplete routes
        for i, player in enumerate(self.players):
            player.finishGame()
            print i, Game.colors[i], player.score

class PublicInfo:
    """
    Data which is known to a particular player, including public information,
    but also the player's own hand and routes.
    """

    def __init__(self, index, game=None):
        self.index = index
        # We accept None as the Game object so that we can construct an empty
        # object to copy into
        if game != None:
            self.graph = game.graph
            self.hand = game.players[index].hand
            self.routes = game.players[index].routes
            print game.players[self.index].hand
            self.handCounts = [sum(p.hand.values()) for p in game.players]
            self.trains = [p.numTrains for p in game.players]
            self.scores = [p.score for p in game.players]
            self.discards = game.discards
            self.revealed = game.revealedTrains

    def deepcopy(self):
        """
        Makes a copy of this object that can be freely modified.
        """
        out = PublicInfo(self.index)
        out.graph = copy.deepcopy(self.graph)
        out.hand = copy.copy(self.hand)
        out.routes = copy.copy(self.routes)
        out.handCounts = copy.copy(self.handCounts)
        out.trains = copy.copy(self.trains)
        out.scores = copy.copy(self.scores)
        out.discards = copy.copy(self.discards)
        out.revealed = copy.deepcopy(self.revealed)
        return out

    def getInvisibleCards(self):
        """
        Builds a dict for the number of each colour of train cards that are
        *not* visible in our own hand or revealed. This includes precisely the
        cards held by other players and the cards in the deck.
        We estimate the distribution of cards in the deck based on this.
        """
        deck = copy.copy(Game.initialDeck)
        for card in self.hand:
            deck[card] -= self.hand[card]
        for card in self.discards:
            deck[card] -= self.discards[card]
        # self.revealed is a list
        for card in self.revealed:
            deck[card] -= 1
        return deck

    def scoreRouteCompletion(self):
        """
        Check whether routes have been completed, and modify scores accordingly.
        Does a BFS from one city on each route, taking only the player's own
        edges, and adds the route value if a path is found to the other city,
        otherwise subtracts.
        """
        for cityA, cityB, points in self.routes:
            found = False
            expanded = set()
            front = [cityA]
            while (len(front) > 0):
                node = front.pop(0)
                if node == cityB:
                    self.scores[self.index] += points
                    found = True
                    break
                if node not in expanded:
                    expanded.add(node)
                    for edge in self.graph.nodes[node]:
                        if edge.owner == self.index:
                            # figure out which end of the edge we were at
                            other = edge.nodes[0]
                            if node == other:
                                other = edge.nodes[1]
                            front.append(other)
            if found == False:
                self.scores[self.index] -= points
            print "%s to %s %s" % (cityA, cityB, "success" if found else "failure")

    def estimateRouteCompletion(self):
        """
        Probably very bad estimate of expected score at the game state
        """
        self.scores[self.index] += sum(self.hand.values())/2
        for cityA, cityB, points in self.routes:
            self.scores[self.index] -= 2*points
            expanded = set()
            front = Queue.PriorityQueue()
            front.put((0, cityA, []))
            while not front.empty():
                cost, node, path = front.get()
                if node == cityB:
                    totalCost = 0
                    totalPaid = 0
                    handCopy = copy.copy(self.hand)
                    handCopy[None] = 0
                    for edge in path:
                        totalCost += edge.price
                        if edge.owner == self.index:
                            totalPaid += edge.price
                    self.scores[self.index] += 3*points*totalPaid/totalCost
                    break
                if node in expanded:
                    continue
                expanded.add(node)
                for edge in self.graph.nodes[node]:
                    if edge.owner == self.index:
                        price = 1
                    elif edge.owner == None:
                        price = edge.price + 1
                    else:
                        continue
                    other = edge.nodes[0]
                    if node == other:
                        other = edge.nodes[1]
                    front.put((cost + price, other, path + [edge]))

class Player:
    """
    Base class for players. To be subclassed to add behaviour.
    """
    def __init__(self, game, index):
        self.game = game # Representation of game
        self.index = index # Identifier for player
        # Initialize hand with 0 of each card before drawing. We access discards
        # because it's the only piece of public information that contains the
        # names of the card colours.
        self.hand = {j: 0 for j in game.discards}
        for j in range(4):
            self.drawTrainCard()
        self.routes = [] # route cards
        self.score = 0 # Current score
        self.numTrains = 45 #8 #45 # Current trains remaining

    def finishGame(self):
        """
        To be overridden by LearningPlayer, which needs to persist weights
        between runs.
        """
        self.scoreRouteCompletion()

    def scoreRouteCompletion(self):
        """
        Shim to avoid duplicating PublicInfo's scoreRouteCompletion logic
        without modifying code that depends on Player having the
        scoreRouteCompletion method.
        """
        publicInfo = PublicInfo(self.index, self.game)
        publicInfo.scoreRouteCompletion()
        self.score = publicInfo.scores[self.index]

    def takeAction(self):
        assert False, "Player has no default takeAction"

    def drawTrainCard(self):
        """
        Draw a card from the deck and add to hand
        """
        self.hand[self.game.drawTrainCard()] += 1

class AttentionDeficitPlayer(Player):
    """
    Looks at edges in random order and plays the first one that it can afford.
    Otherwise, draws randomly.
    Contains a lot of the helper methods used by other AI players.
    """

    def pickRouteCards(self, numDiscard):
        """
        Minimum viable route drawing.
        """
        self.routes += self.game.drawRouteCards()

    def takeAction(self):
        """
        Plays edge if possible, otherwise draws trains
        """

        # Doesn't pre-filter edges before trying them
        if self.tryEdges(self.game.graph.edges):
            return

        # We don't do any searches, so we don't have any priorities here
        self.drawTrains([])

    def drawTrains(self, chosenEdges):
        """
        Minimum viable train drawing. Draws two cards at random.
        """
        self.drawTrainCard()
        self.drawTrainCard()

    def tryEdges(self, chosenEdges):
        """
        Attempt to buy edges in the order specified by keyEdges.
        Returns the edge (a true value) if successful.
        """
        for edge in sorted(chosenEdges, key = self.keyEdges):
            if self.tryPlay(edge):
                print edge
                return edge

    def keyEdges(self, edge):
        """
        Minimum viable edge ordering.
        """
        return random.random()

    def tryPlay(self, edge):
        """
        Checks if this player can afford the given edge, and takes it,
        returning True, if possible; otherwise returns False.
        Always uses minimum possible number of rainbows.
        """

        # fails if the edge is already owned, or if we don't have enough trains
        # to take it
        if edge.owner != None or edge.price > self.numTrains:
            return False

        # Number of colored cards needed.
        minimum = edge.price - self.hand["rainbow"]
        color = edge.color
        if color:
            if self.hand[color] < minimum:
                return False
        else:
            # Edge is grey, so we pick a random color that we have enough of.
            ok = [key for key in self.hand if self.hand[key] >= minimum and key != "rainbow"]
            if len(ok) <= 0:
                return False
            color = random.choice(ok)

        self.consumeTrains(color, edge.price)
        # Set ownership
        edge.owner = self.index
        return True

    def consumeTrains(self, color, num):
        """
        Discards an appropriate number of train cards from the hand of the
        color given, making up the difference in rainbows. Removes trains from
        the player.
        Assumes that the caller has already checked that the number of rainbows
        is sufficient to cover overflow.
        """

        self.score += Game.scoring[num]
        self.numTrains -= num
        if self.hand[color] >= num:
            self.game.discards[color] += num
            self.hand[color] -= num
        else:
            self.game.discards[color] += self.hand[color]
            num -= self.hand[color]
            self.hand[color] = 0
            self.game.discards["rainbow"] += num
            self.hand["rainbow"] -= num

class PatientPlayer(AttentionDeficitPlayer):
    """
    Always tries to play the most expensive edges first, but is not correctly
    implemented.
    """

    def keyEdges(self, edge):
        return -edge.price

class GreedyPlayer(AttentionDeficitPlayer):
    """
    Player which attempts to play the edges along the cheapest paths for its
    routes, and falls back to AttentionDeficitPlayer when all routes are
    complete or uncompletable.
    """

    def ucsRoute(self, route, chosenEdges):
        """
        Finds a cheapest path for the route given, assuming that edges already
        owned and chosenEdges are free
        """
        cityA, cityB, _ = route
        expanded = set()
        front = Queue.PriorityQueue()
        front.put((0, cityA, []))
        while (not front.empty()):
            cost, node, path = front.get()
            if node == cityB:
                return (cost, path)
            if node in expanded:
                continue
            expanded.add(node)
            for edge in self.game.graph.nodes[node]:
                # figure out which side of the edge we're at
                other = edge.nodes[0]
                if node == other:
                    other = edge.nodes[1]

                if edge.owner == self.index or edge in chosenEdges:
                    front.put((cost, other, path))
                elif edge.owner == None:
                    front.put((cost + edge.price, other, path + [edge]))

        # No path found, so we consider this case infinitely expensive
        return (float("inf"), None)

    def chooseEdges(self):
        """
        Find a set of edges that completes the routes held, in order drawn.
        Edges used for previous routes are not charged for subsequent routes.
        Basically a greedy algorithm at the route level, does not necessarily
        generate the optimal edge set.
        """
        chosenEdges = []
        for route in self.routes:
            _, edges = self.ucsRoute(route, chosenEdges)
            if edges != None:
                chosenEdges += edges
        return chosenEdges

    def takeAction(self):
        """
        Called by game. Skeleton action which calls other instance methods,
        which are overriden by subclasses, for particular behavior.
        """

        chosenEdges = self.chooseEdges()
        print map(str, chosenEdges)

        if len(chosenEdges) == 0:
            self.noWantedEdges()
            return

        if self.tryEdges(chosenEdges):
            return

        print "drawing cards"
        self.drawTrains(chosenEdges)

    def noWantedEdges(self):
        """
        Fallback when no edges are desired for route completion.
        """
        print "\t\t\t\t\tATTENTION DEFICIENCY (GREEDY)"
        AttentionDeficitPlayer.takeAction(self)

class ColorVisionPlayer(GreedyPlayer):
    """
    Experimental player with awareness of the colors of edges and cards in
    hand, and prioritizes edges that seem to be affordable.
    """
    def ucsRoute(self, route, chosenEdges):
        cityA, cityB, _ = route
        expanded = set()
        front = Queue.PriorityQueue()
        front.put((0, cityA, [], {j.color: 0 for j in self.game.graph.edges}))
        while (not front.empty()):
            cost, node, path, prices = front.get()
            if node == cityB:
                return (cost, path)
            if node in expanded:
                continue
            expanded.add(node)
            for edge in self.game.graph.nodes[node]:
                other = edge.nodes[0]
                if node == other:
                    other = edge.nodes[1]

                if edge.owner == self.index or edge in chosenEdges:
                    front.put((cost, other, path, prices))
                elif edge.owner == None:
                    newPrices = copy.copy(prices)
                    newPrices[edge.color] += edge.price
                    # computes path cost using computeCost
                    front.put((self.computeCost(newPrices, len(path)+1), other, path + [edge], newPrices))

        return (float("inf"), None)

    def computeCost(self, prices, length):
        """
        Estimates a relative cost for a path, given its length and the total
        colors needed.
        """
        grey = prices[None]
        prices = {j: prices[j] - self.hand[j] for j in prices if j}
        return 2*length + grey + 4*max(prices[j] for j in prices)

    def keyEdges(self, edge):
        """
        Orders edges by price, counting colored edges as more expensive so that
        we do not spend desired colored cards on grey edges.
        """
        price = edge.price
        if edge.color:
            price += 2
        return -price

class RoutePlayer(GreedyPlayer):
    """
    Variant of GreedyPlayer with the ability to draw more route cards and
    discard some of them.
    """

    def pickRouteCards(self, numDiscard):
        """
        Draws route cards, then keeps only most easily achieved option.
        """

        # No behaviour change at beginning of game
        if numDiscard == 1:
            AttentionDeficitPlayer.pickRouteCards(self, 1)
            return

        # numDiscard == 2 when drawing as an action
        drawn = self.game.drawRouteCards()
        print drawn
        picked = []
        while len(drawn) > numDiscard:
            ucsList = map(lambda r: self.ucsRoute(r, []), drawn)
            best = min(range(len(drawn)), key=lambda i: ucsList[i])
            picked.append(drawn.pop(best))
        print picked
        print drawn
        self.routes += picked
        self.game.putRouteCards(drawn)

    def noWantedEdges(self):
        """
        Tries to draw routes if it's done with its routes, as long as there
        seem to be enough turns left in the game.
        """
        if min(p.numTrains for p in self.game.players) > 12 and len(self.game.routeDeck) > 3:
            self.pickRouteCards(2)
            return

        # fallback
        print "\t\t\t\t\tATTENTION DEFICIENCY (ROUTE)"
        print self.numTrains
        AttentionDeficitPlayer.takeAction(self)

class FaceUpPlayer(RoutePlayer):
    """
    Variant of RoutePlayer which draws face-up from the revealed trains when
    visible cards would help complete routes.
    """

    def drawFaceUp(self, index):
        """
        Helper to pick up a revealed card at given index.
        """
        color = game.revealedTrains[index]
        self.hand[color] += 1
        game.revealedTrains[index] = None
        game.revealTrainCards()
        return color

    def drawTrains(self, chosenEdges):
        """
        Identifies which colors are wanted, and tries to pick them up from the
        revealed trains if present. Falls back to random drawing otherwise.
        """

        # Find the colored costs of desired edges
        desiredColors = {j.color: 0 for j in self.game.graph.edges}
        for edge in chosenEdges:
            desiredColors[edge.color] += edge.price

        # Finds the deficiency for each color
        for key in desiredColors:
            if key != None:
                desiredColors[key] -= self.hand[key]

        print self.routes

        wanted = desiredColors.keys()

        for i in range(2):
            # sort the set of colors in order of desire
            wanted.sort(key = lambda k: -desiredColors[k])
            print "the wants of %d self.index are " % self.index + str([(c, desiredColors[c]) for c in wanted])
            index = None

            # Goes through wanted colors until we find a available color that
            # we want, or we don't lack any of the remaining colors.
            for color in wanted:
                if desiredColors[color] <= 0:
                    break
                indices = [j for j, train in enumerate(self.game.revealedTrains) if color == train]
                # Choose one at random if there are multiple. Doesn't really matter.
                if len(indices) > 0:
                    index = random.choice(indices)
                    break

            if index != None:
                # Found a color we want
                print "drawing face-up %s (wanted)" % color
                desiredColors[color] -= 1
                self.drawFaceUp(index)
            elif desiredColors[None] > 0:
                # We need a lot of greys, so we try to build stacks of cards of the same color
                choice = max(self.game.revealedTrains, key = lambda k: self.hand[k] if k != "rainbow" else -1)
                print "drawing face-up %s (stacking)" % choice
                print self.game.revealedTrains
                desiredColors[choice] -= 1
                self.drawFaceUp(self.game.revealedTrains.index(choice))
            elif i == 0 and len([k for k in desiredColors.values() if k > 0]) <= 3 and "rainbow" in self.game.revealedTrains:
                # Takes visible rainbow if possible if we only want a few specific colors
                print "drawing face-up rainbow"
                self.drawFaceUp(self.game.revealedTrains.index("rainbow"))
                # We can only draw one rainbow
                break
            else:
                print self.game.revealedTrains
                print "nothing was wanted, drawing randomly"
                self.drawTrainCard()

class LearningPlayer(FaceUpPlayer):
    """
    Player which chooses between higher-order actions (draw/route/specific
    edge) using Q-learning. The specifics of train and route drawing are
    decided by logic inherited from RoutePlayer and FaceUpPlayer.
    """

    def __init__(self, game, index):
        FaceUpPlayer.__init__(self, game, index)
        # Q-learning parameters
        self.alpha = 0.05
        self.epsilon = 0.3
        self.discount = 0.9
        self.prevQValue = 0.0
        self.prevFeatures = {}
        self.weights = util.Counter()

        # Reload weights from learning.txt. Starts with zeroes otherwise.
        try:
            with open("learning.txt") as learning:
                self.episodesSoFar = int(learning.readline())
                for weight in learning.readlines():
                    name, value = weight.split()
                    self.weights[name] = float(value)
        except IOError:
            self.episodesSoFar = 0
            print "no learning"
        print self.weights

    def getQValue(self, action):
        """
        Computes Q-value for action from current state.
        """
        total = 0
        features = self.getFeatures(action)
        for key in features:
            total += self.weights[key] * features[key]
        return total

    def getValue(self):
        """
        Compute maximum Q-value for legal actions from current state.
        """
        maxVal = -float("inf")
        actions = self.getLegalActions()
        for act in actions:
            temp = self.getQValue(act)
            if temp > maxVal:
                maxVal = temp
        return maxVal if maxVal != -float("inf") else 0.0

    def getAction(self):
        """
        Choose next action, either randomly or by best Q value, depending on
        epsilon.
        """
        legalActions = self.getLegalActions()
        assert len(legalActions)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues()

    def computeActionFromQValues(self):
        """
        Choose action with best Q value.
        """
        bestActions = []
        maxVal = -float("inf")
        actions = self.getLegalActions()
        assert len(actions) > 0
        for act in actions:
            qVal = self.getQValue(act)
            if qVal == maxVal:
                bestActions.append(act)
            elif qVal > maxVal:
                bestActions = [act]
                maxVal = qVal
        return random.choice(bestActions)

    def takeAction(self):
        """
        Determine preferred action, and then perform it.
        """

        # We update every time the state changes, using saved information about
        # the previous state.
        self.update(0)

        action = self.getAction()
        if action[0] == "draw":
            self.drawTrains(self.chooseEdges())
        elif action[0] == "route":
            self.pickRouteCards(2)
        else:
            # we cannot reuse code for this, because getLegalActions gives a
            # distinct action for each edge * color pair.
            assert action[0] == "edge"
            _, edge, color = action
            if edge.price <= self.hand[color]:
                self.hand[color] -= edge.price
                self.game.discards[color] += edge.price
            else:
                rainbows = edge.price - self.hand[color]
                self.game.discards[color] += self.hand[color]
                self.hand[color] = 0
                self.game.discards["rainbow"] += rainbows
                self.hand["rainbow"] -= rainbows
            assert edge.owner == None
            edge.owner = self.index
            self.score += Game.scoring[edge.price]
            self.numTrains -= edge.price

        # Save information about this state for next update, which must occur
        # after opponents go.
        self.prevQValue = self.getQValue(action)
        self.prevFeatures = self.getFeatures(action)
        print "I AM ACTION", action

    def getFeatures(self, action):
        """
        Computes a set of features for the state after taking the given action.
        """
        features = {}

        # Overall bias, as well as bias for each action
        features["bias"] = 1.0
        features["bias." + action[0]] = 1.0

        # Scale the number of trains between 0 and 1 by dividing by the maximum
        features["trainsRemainingRatio"] = self.numTrains/45.0
        features["sketchyMinTrainRatio"] = min(p.numTrains for p in self.game.players)/45.0

        # Scale number of cards by dividing by maximum
        features["sketchyTotalHandRatio"] = sum(len(p.hand) for p in self.game.players)/110.0
        myHandCount = sum(self.hand.values())

        # Computes status of routes after this action by performing a search
        # for each route.

        incompleteRoutes = 0.0
        totalRoutes = len(self.routes)
        incompleteRouteStake = 0.0
        totalRouteStake = 0.0

        # Assume the edge has been taken if we are taking an edge
        allNeededEdges = []
        if action[0] == "edge":
            allNeededEdges.append(action[1])

        # Accumula
        totalRouteCost = 0.0
        totalRouteEdgesNeeded = 0.0
        minEndpointDegree = float("inf")
        totalCost = 0.0

        for route in self.routes:
            totalRouteStake += route[2]
            for city in route[0:2]:
                degree = 0
                for edge in self.game.graph.nodes[city]:
                    if edge.owner == self.index:
                        degree = 0 # will get ignored
                        break
                    elif edge.owner == None:
                        degree += 1
                if degree > 0:
                    minEndpointDegree = min(degree, minEndpointDegree)
            cost, edges = self.ucsRoute(route, allNeededEdges)
            if edges != []:
                incompleteRoutes += 1
                incompleteRouteStake += route[2]
            if edges != None:
                allNeededEdges += edges
                totalRouteCost += cost
                totalRouteEdgesNeeded += len(edges)

        # Necessary (but not sufficient) condition for us to be able to afford
        # all the edges on our routes.
        features["extremelySketchyAffordability"] = 1.0 if myHandCount >= totalRouteCost else 0.0

        # Computes a rough estimate of affordability which is aware of colors.
        handCopy = copy.copy(self.hand)
        if action[0] == "edge":
            handCopy[action[2]] -= edge.price
        handCopy[None] = 0
        allNeededEdges.sort(reverse=True, key=lambda e: e.color)
        for edge in allNeededEdges:
            handCopy[edge.color] -= edge.price
            if edge.color != None and handCopy[edge.color] < 0:
                handCopy["rainbow"] += handCopy[edge.color]
                handCopy[edge.color] = 0
        if action[0] == "draw":
            handCopy[None] += 2
        features["affordability"] = 1.0/(1 - min(sum(handCopy.values()), 0) - min(handCopy["rainbow"], 0))

        if action[0] == "draw":
            # We get more cards if we draw them
            myHandCount += 2
            features["sketchyTotalHandRatio"] += 2/110.0
        elif action[0] == "route":
            # Assumes close to worst-case when drawing routes, because we don't
            # have much information about the route deck
            incompleteRoutes += 1
            totalRoutes += 1
            incompleteRouteStake += 20
            totalRouteStake += 20
        else:
            # updates information about trains and hand size after playing edge
            assert action[0] == "edge"
            _, edge, color = action
            features["trainsRemainingRatio"] -= edge.price/45.0
            myHandCount -= edge.price
            features["sketchyTotalHandRatio"] -= edge.price/110.0
            features["sketchyMinTrainRatio"] = min(features["sketchyMinTrainRatio"], features["trainsRemainingRatio"])

        # Scaled hand size
        features["myHandRatio"] = myHandCount/110.0
        # Fraction of routes not completed
        features["routeIncompletionRatio"] = float(incompleteRoutes) / totalRoutes
        # Scaled number of routes held
        features["routesFromDeckRatio"] = totalRoutes/30.0
        # Estimate of "crowdedness" of route endpoints
        features["maxInverseEndpointDegree"] = 1.0/minEndpointDegree
        # Strangely scaled total cost of edges on route
        features["inverseRouteCost"] = 1.0/(1+totalRouteCost)
        # Fraction of route points which is still at risk due to incompleteness
        features["incompleteRouteFractionOfRoutesPoints"] = incompleteRouteStake / totalRouteStake

        return features

    def update(self, reward):
        #print "UPTGATING WEGTGTGS"
        #called by observeTransition
        currentValue = self.getValue()
        correction = reward + self.discount * currentValue - self.prevQValue
        print "REWARD", reward
        print "SECOND TERM", self.discount * currentValue
        print "PREVIOUS", self.prevQValue
        print "CORRECTION", correction
        for key in self.prevFeatures:
            adjustment = self.alpha * correction * self.prevFeatures[key]
            print "ADJUSTMENT AT", key, adjustment, "FROM", self.weights[key], "TO", self.weights[key] + adjustment
            self.weights[key] += adjustment

    def getLegalActions(self):
        """
        Returns a list of legal actions for getAction.
        """

        actions = []

        # Only allow drawing if there are enough cards to draw
        if sum(self.game.trainDeck.values()) + sum(self.game.discards.values()) >= 2:
            actions.append(("draw",))

        # Only allow drawing if there are enough cards to draw
        if len(self.game.routeDeck) >= 3:
            actions.append(("route",))

        # Produces an action for each playable combination of edge and color
        for j, edge in enumerate(self.game.graph.edges):
            # skip if already owned or not enough trains to play
            if edge.owner != None or edge.price > self.numTrains:
                continue

            # Find colors which can be played on the edge
            if edge.color != None:
                colors = [edge.color]
            else:
                colors = [j for j in self.hand.keys() if j != "rainbow"]
            # Add only colors which we have enough of to play the edge
            actions += [("edge", edge, j) for j in colors if self.hand[j] + self.hand["rainbow"] >= edge.price]

        return actions

    def stopEpisode(self):
        """
        Learning episode is done, write out weights for use in next episode.
        """
        with open("learning.txt", "w") as learning:
            learning.write(str(self.episodesSoFar+1) + "\n")
            for key in self.weights:
                learning.write(key + " " + str(self.weights[key]) + "\n")

    def finishGame(self):
        """
        Override to save the final weights.
        """
        self.scoreRouteCompletion()
        self.update(self.score)
        self.stopEpisode()

class LearnedPlayer(LearningPlayer):
    """
    Functions as LearningPlayer, but doesn't learn or explore.
    """

    def __init__(self, game, index):
        LearningPlayer.__init__(self, game, index)
        # Don't explore
        self.epsilon = 0.0

    def update(self, reward):
        # Don't learn
        pass

    def stopEpisode(self):
        # Don't write to file
        pass

class MonteCarloPlayer(AttentionDeficitPlayer):
    def monteCarloTreeSearch(self, n, game):
        """
        Creates the MonteNode and calls sample for n iterations.
        After sampling the node, the most optimal action that
        leads to the child with best rewards is selected.
        If no child is created during the sampling, a random
        action is taken instead.
        """
        root = MonteNode(None, PublicInfo(self.index, game))
        # root is sampled.
        for i in range(0, n):
            root.sample(game)

        maxReward = -99999999999999999999
        selectedChild = None
        # best child is selected
        for child in root.children:
            if child.reward >= maxReward:
                selectedChild = child
                maxReward = child.reward
        if selectedChild != None and selectedChild.action != None:
            return selectedChild.action
        # return random action if no child.
        return random.choice(root.getLegalActions(root.det))

    def drawUp(self, index):
        """
        Remove a face up card and add it to hand.
        Replaces taken cards with cards from deck.
        The second card taken should not be rainbow.
        Returns the color of the card taken.
        """
        print index
        color = game.revealedTrains[index]
        self.hand[color] += 1
        game.revealedTrains[index] = None
        game.revealTrainCards()

    def drawFakeUp(self, index):
        """
        A version for the MonteCarloTreeSearch's simulated playout
        """
        color = self.game.revealedTrains[index]
        self.hand[color] += 1
        self.game.revealedTrains[index] = None
        self.game.revealFakeTrainCards()

    def takeAction(self, givenAction = None):
        """
        takes action for the game when not passed an action argument. If
        passed an action argument, does a simulated version of an action
        that changes the determinization.
        """
        action = None
        # if no action passed as an argument, (which means this MonteCarloPlayer is the real one),
        # do MCTS
        if givenAction == None:
            action = self.monteCarloTreeSearch(75, self.game)
        else:
            # if action given (simulated player), action = givenAction
            action = givenAction
        if action[0] == "edge":
            _, rainbows, color, edgeIx = action
            edge = self.game.graph.edges[edgeIx]
            assert edge.color == None or edge.color == color
            edge.owner = self.index
            used = edge.price - rainbows
            assert self.hand[color] >= used
            self.game.discards[color] += used
            self.hand[color] -= used
            assert self.hand["rainbow"] >= rainbows
            self.game.discards["rainbow"] += rainbows
            self.hand["rainbow"] -= rainbows
            self.score += Game.scoring[edge.price]
            self.numTrains -= edge.price
        elif action[0] == "drawDeck":
            # draw from fake train deck (a list) if player is simulated.
            if givenAction != None:
                self.hand[self.game.drawFakeTrainCard()] += 1
                self.hand[self.game.drawFakeTrainCard()] += 1
            else:
                # draw from real train deck (a dict) if player is real.
                self.hand[self.game.drawTrainCard()] += 1
                self.hand[self.game.drawTrainCard()] += 1
        elif action[0] == "routes":
            self.tempRoutes = []
            drawnRoutes = self.game.drawRouteCards()
            for route in drawnRoutes:
                self.tempRoutes.append(route)
            # discards route cards that are not selected according to the action index
            for i in range (0, 3):
                if i != action[2]:
                    self.game.routeDeck.append(self.tempRoutes[i])
                    self.tempRoutes[i] = None
            # appends the route card to the player's route cards
            for route in self.tempRoutes:
                if route != None:
                    self.routes.append(route)
        elif action[0] == "drawPile":
            if givenAction != None:
                # draws from the simulated face-up pile which is updated by the simulated train deck
                self.drawFakeUp(action[1])
            else:
                # draws from the real face-up pile
                self.drawUp(action[1])
            # checks to see if second action is a string
            if not isinstance(action[2],basestring):
                if action[2] == 5:
                    # draws from deck
                    if givenAction != None:
                        self.hand[self.game.drawFakeTrainCard()] += 1
                    else:
                        self.hand[self.game.drawTrainCard()] += 1
                else:
                    # draws from pile
                    if action[2] != None:
                        if givenAction != None:
                            self.drawFakeUp(action[2])
                        else:
                            self.drawUp(action[2])
        else:
            assert False, "unknown action " + action[0]

class MonteNode:
    def __init__(self, action, publicInfo, det = None, parent = None):
        self.children = [] #children for actions we have simulated
        self.action = action #action that got us to this node
        self.visits = 0 #number of times we have gotten to this state
        self.availability = 1 #number of times we have gotten to this state's parent
        self.det = det #the determinization associated with the node
        self.reward = 0 #back-propogated from evaluation function/terminal state
        self.publicInfo = publicInfo #its known state of the game (minus the hidden state)
        self.expanded = False #boolean vaolue to see whether it has expanded or not
        self.parent = parent #its parent

    def getLegalActions(self, det):
        actionList = []
        highestEdgeIndex = 100
        with open("colors.txt") as colors:
            colorList = [c.split()[0] for c in colors]
        for i in range(0,det.players[self.publicInfo.index].hand["rainbow"] + 1):
            for color in colorList:
                if color != "rainbow":
                    for j in range (0, highestEdgeIndex):
                        # checks that the conditions to play the edges are met: sufficient number of colored train cards, rainbow train cards and color of the edge should match the cards being used.
                        if det.graph.edges[j].owner == None and (det.players[self.publicInfo.index].hand[color] - i >= det.graph.edges[j].price) and (det.graph.edges[j].color == color or det.graph.edges[j].color == None):
                            actionList.append(("edge", i, color, j))
        # generates all possible combinations of route cards.
        if (len(det.routeDeck) > 3 ):
            for i in range(0,3):
                for j in range(0,3):
                    if i != j and i <= j:
                        for k in range(0, 3):
                            if k != i and k != j:
                                actionList.append(("routes",det.routeDeck[k],k))

        #add draw option
        someDeck = det.fakeTrainDeck
        total = len(someDeck)
        # if size of deck is less than or equal to 1, return action list immediately without adding drawing legal actions
        if total <= 1:
            return actionList
        # adds drawing the top 2 cards as a legal action
        card1 = someDeck[0]
        card2 = someDeck[1]
        actionList.append(("drawDeck",card1,card2))
        if total <= 10:
            return actionList
        index1 = None
        index2 = None
        # adds all possible combinations of draw from face-up pile actions.
        for i in range (0, 5):
            for j in range (0, 5):
                # if first card is rainbow then only update index1, which makes the agent only draw the rainbow card
                if det.revealedTrains[i] == "rainbow":
                    index1 = i
                else:
                    # if first and second cards are not rainbows, then add draw 2 cards from the face-up pile as an action
                    index1 = i
                    if det.revealedTrains[j] != "rainbow":
                        index2 = j
                    else:
                        # if first is not rainbow but second card is, then add draw 1 card from the face-up pile and draw 1 from deck as an action
                        index2 = someDeck[0]
                actionList.append(("drawPile",index1,index2))

        return actionList

    def generateDeterminization(self, game, publicInfo, playerIndex):
        """
        Generates a determinization. The determinization is a game state from a certain player's perspective.
        The determinization contains info on the public game state and also a randomly generated guess on what the
        hidden state of the train and route deck could be.
        """
        det = Game()
        det.insertPlayers(copy.deepcopy(game.players))
        det.graph = copy.deepcopy(publicInfo.graph)
        det.players[publicInfo.index].routes = game.players[publicInfo.index].routes
        # gets all the hidden train cards.
        invisibleDeck = publicInfo.getInvisibleCards()
        someHandCounts = []
        someTrainDeck = []
        # remembers the size of all players' hands
        for handIndex in publicInfo.handCounts:
            someHandCounts.append(handIndex)
        for i, player in enumerate(det.players):
            if i != publicInfo.index:
                det.players[i].hand = det.players[i].hand.fromkeys(det.players[i].hand,0)

        invisibleDeck = publicInfo.getInvisibleCards()
        invisibleList = list(invisibleDeck)
        # generates a guess of the train deck and opponent players' hand
        while (sum(invisibleDeck.values()) != 0 ):
            handReceivedCard = False
            randomCard = random.choice(invisibleList)
            # gets a random card in the pool of hidden train cards
            while invisibleDeck[randomCard] == 0 and sum(invisibleDeck.values()) != 0:
                randomCard = random.choice(invisibleList)
            # adds random card to one of the player's hands
            for handIndex, handNumber in enumerate(someHandCounts):
                if handIndex != playerIndex and handNumber > 0 and handReceivedCard == False:
                    det.players[handIndex].hand[randomCard] += 1
                    someHandCounts[handIndex] -= 1
                    handReceivedCard = True
            # if player's hands are filled, append it to the guessed train deck.
            if handReceivedCard == False:
                someTrainDeck.append(randomCard)
            invisibleDeck[randomCard] -= 1
        # determinization holds the guessed train deck.
        det.fakeTrainDeck = someTrainDeck
        # shuffle the train deck to randomize it.
        random.shuffle(det.fakeTrainDeck)
        # copy the rest of the public state.
        det.revealedTrains = copy.deepcopy(publicInfo.revealed)
        det.discards = copy.deepcopy(publicInfo.discards)
        for player in det.players:
            player.game = det
        # face-up cards
        self.det = det
        return det

    def getSuccessorState(self, action, det, node):
        """
        Gets the MonteNode's successor state.
        """
        successor = copy.deepcopy(self.publicInfo)
        #action indexes represent different things
        # action[0] ~ choose to draw train card, or draw route card, or play trains
        # - "edge":
        #    action[1] ~ number of rainbow cards used
        #    action[2] ~ color of train to be played
        #    action[3] ~ index of edge to occupy
        # - "draw"
        #    action[1:2] ~ colors of cards, action[1] <= action[2]
        # - "routes"
        #    action[1] ~ sorted list of routes

        if action[0] == "edge":
            print action, "ACTION"
            #asserts that the conditions are met for the action (e.g: sufficient number of train cards, edge has no owner, handsize is correct)
            _, rainbows, color, edgeIx = action
            edge = successor.graph.edges[edgeIx]
            assert edge.color == None or edge.color == color
            edge.owner = successor.index
            used = edge.price - rainbows
            # checks that hand has enough cards of a certain train colour
            assert successor.hand[color] >= used
            # discards the used cards
            successor.discards[color] += used
            successor.hand[color] -= used
            # checks that rainbow cards in hand are more than or equal to the rainbow cards to be used.
            assert successor.hand["rainbow"] >= rainbows
            # discards used rainbow cards
            successor.discards["rainbow"] += rainbows
            successor.hand["rainbow"] -= rainbows
            # reduces hand size
            successor.handCounts[self.publicInfo.index] -= edge.price

            assert successor.handCounts[self.publicInfo.index] == sum(successor.hand.values())
            # increases the successor's scores.
            successor.scores[self.publicInfo.index] += Game.scoring[edge.price]
            successor.trains[self.publicInfo.index] -= edge.price
        elif action[0] == "drawDeck":
            # draws
            _, card1, card2 = action
            #assert card1 <= card2
            # adds drawn cards to the hand
            successor.hand[card1] += 1
            successor.hand[card2] += 1
            successor.handCounts[self.publicInfo.index] += 2
            assert successor.handCounts[self.publicInfo.index] == sum(successor.hand.values())
        elif action[0] == "drawPile":
            _, card1, card2 = action
            # draws from the face-up pile
            successor.hand[successor.revealed[card1]] += 1
            # if first card drawn is rainbow, draw nothing else
            if successor.revealed[card1] == "rainbow":
                successor.handCounts[self.publicInfo.index] -= 1
            # if 2nd card is not a string, draw from face-up again and add it to hand
            elif card2 != None and not isinstance(card2, basestring):
                successor.hand[successor.revealed[card2]] += 1
            else:
                # if 2nd card is a string (drawn from deck), add it to hand
                successor.hand[card2] += 1
            successor.handCounts[self.publicInfo.index] += 2
        elif action[0] == "routes":
            #assert action[1] == sorted(action[1])
            successor.routes += action[1]
        else:
            assert False, "unknown action " + action[0]
        return MonteNode(action, successor, det, node)

    def sample(self, game):
        """
        The main layout of the MCTS. Sample generates a determinization and
        then selects randomly selects a child at the frontier using the UCB formula compatible with the determinization.
        It expands that child (and updates the corresponding determinization and public information class) and simulates
        random playouts on the updated determinization to calculate estimated reward of the simulated playthrough. The
        reward is then backpropagated through the path of selected nodes.
        """
        det = self.generateDeterminization(game, self.publicInfo, self.publicInfo.index)
        self.det = det
        print "DETERMINIZATION GENERATED"
        nextState = None
        winner = None
        # the tree of nodes is traversed downwards until a child that still has unexpanded nodes corresponding to the determinization is generated.
        nextState = self.select(det)
        endGame = False
        # check if game has ended
        for player in nextState[1].players:
            if player.numTrains <= 3:
                endGame = True
        if endGame == False:
            # if game has not ended then expands the node at the frontier
            nextState = self.expand(nextState[0],nextState[1])
        detCopy = copy.deepcopy(nextState[1])
        # the new determinization created from expansion is used to do random playouts
        reward = self.randomPlayout(detCopy, 0)
        print '\n COMPLETE RANDOMPLAYOUT \n'
        # estimated reward is then backpropagated up the path of nodes traversed.
        self.updateValue(reward, nextState[0])
        print '\n COMPLETE UPDATE VALUE \n'

    def select(self, det):
        """
        The tree of nodes is randomly traversed using a UCT formula (Upper confidence bound applied to trees) to get to a child
        that still has an action compatible to the determinization that has not been explored yet.
        """
        weights = []
        oldDet = copy.deepcopy(det)
        someDet = None
        selectedNode = self
        selectedNode.det = oldDet
        endGame = False
        foundChild = False
        # gets the list of legal actionso f the old determinization.
        actionList = self.getLegalActions(oldDet)
        # checks if endgame is reached yet.
        for player in det.players:
            if player.numTrains <= 3:
                endGame = True
        # checks if node has expanded a child at least once and that endgame is not true.
        while selectedNode.expanded == True and endGame == False:
            for action in actionList:
                for child in selectedNode.children:
                    # compares the action to the actions used to lead the selected node to its children. Also checks if the action is compatible to the current determinization (by comparing
                    # number of cards in hand and discard pile between the determinization and the child's public game state.)
                    someDet = copy.deepcopy(oldDet)
                    someDet.players[self.publicInfo.index].takeAction(child.action)
                    if child.action == action and cmp(child.publicInfo.hand, someDet.players[self.publicInfo.index].hand) == 0 and cmp(child.publicInfo.discards, someDet.discards) == 0:
                        # if action that led to child is the same as the selected action, increase child availability, that is the count in which the child is available as an option for the current action, by 1.
                        child.availability += 1
                        # the UCT formula (child's reward/child's visit count) + k * square root ( ln(child's availability)/ child's visits) is used to calculate the child's weight
                        w = (child.reward/child.visits) + 7 * math.sqrt(math.log(child.availability) / child.visits)
                        weights.append((w, child))
                        foundChild = True
                        break
                    else:
                        foundChild = False
                if foundChild == False:
                    break
            # if there is no child that corresponds to that current action (that is, the currently selected node has an action of that determinization that has not been expanded yet),
            # the currently selected node and its corresponding determinization are returned instead.
            if foundChild == False:
                return (selectedNode, selectedNode.det)
            maxI = -9999
            # if all the current actions of the determinization have been explored before, then we pick a child that has a maximum weight (calculated using the UCT formula).
            # That child becomes a new selected node.
            for i, child in weights:
                if i >= maxI:
                    selectedNode = child
                    selectedNode.availability -= 1
                    maxI = i
            # generates a new determinization using the child's action.
            newDet = copy.deepcopy(oldDet)
            selectedNode.det = newDet
            selectedNode.det.players[self.publicInfo.index].takeAction(selectedNode.action)
            actionList = selectedNode.getLegalActions(selectedNode.det)
            oldDet = newDet
            # checks if endgame has been reached  yet.
            for player in selectedNode.det.players:
                if player.numTrains <= 3:
                    endGame = True
        return (selectedNode, selectedNode.det)

    # generates a random playout up to "depth" number of times using the determinization that is passed through as an argument.
    def randomPlayout(self, det, depth):
        evaluate = False
        endGame = False
        for player in det.players:
            if player.numTrains <= 3:
                endGame = True
            elif depth >= 40:
                evaluate = True
        # if the simulated game has ended, return the player's score
        if endGame:
            det.players[self.publicInfo.index].scoreRouteCompletion()
            return det.players[self.publicInfo.index].score
        # if depth has been reached, return an estimated player's score
        elif evaluate:
            det.estimateRouteCompletion(self.publicInfo.index)
            score = det.players[self.publicInfo.index].score
            return score
        else:
            # update the determinization using a randomly selected action
            actionList = self.getLegalActions(det)
            # if no legal actions possible, return an estimate score route completion
            if len(actionList) == 0:
                det.estimateRouteCompletion(self.publicInfo.index)
                return det.players[self.publicInfo.index].score
            chosenAction = random.choice(actionList)
            det.players[self.publicInfo.index].takeAction(chosenAction)
            return self.randomPlayout(det, depth + 1)

    def randomAction(self):
        return random.choice(self.actions)

    def expand(self, node, det):
        """
        Expands the node.
        """
        node.expanded = True
        # gets a random action from the current determinization. The node is expanded and the determinization
        # is updated using the action.
        actionList = node.getLegalActions(det)
        randomAct = random.choice(actionList)
        newDet = copy.deepcopy(det)
        newDet.players[self.publicInfo.index].takeAction(randomAct)
        newNode = node.getSuccessorState(randomAct, newDet, node)
        node.children.append(newNode)
        return (newNode,newDet)

    def updateValue(self, newScore, node):
        """
        Updates the reward of the path of nodes traversed.
        """
        while node.parent != None:
            node.visits += 1
            node.reward += newScore
            actionList = node.getLegalActions(node.det)
            node = node.parent
        self.reward = max(newScore, self.reward)

class ShoddyUIPlayer(Player):
    """
    Player which asks for user input for each action.
    Assumes the user gives correct input, and is likely to crash otherwise.
    """

    def evilInput(self, prompt):
        """
        Input method which accepts magic prefixes to execute commands for
        debugging.
        """
        while True:
            try:
                got = raw_input(prompt)
                if got == "":
                    pass
                elif got[0] == "=":
                    print eval(got[1:])
                elif got[0] == "!":
                    exec(got[1:])
                else:
                    return got
            except Exception as e:
                print e

    def drawFaceUp(self, index, time):
        """
        Remove a face up card and add it to hand.
        Replaces taken cards with cards from deck.
        The second card taken should not be rainbow.
        Returns the color of the card taken.
        """
        color = game.revealedTrains[index]
        if time == 1:
            if color == "rainbow":
                index = int(self.evilInput("different index "))
                self.drawFaceUp(index, time)
        self.hand[color] += 1
        game.revealedTrains[index] = None
        game.revealTrainCards()
        return color

    def drawRouteCards(self):
        """
        Draws route cards and puts in a temporary space to decide which to discard
        """
        self.tempRoutes = []
        drawnRoutes = self.game.drawRouteCards()
        for route in drawnRoutes:
            self.tempRoutes.append(route)

    def pickRouteCards(self, numDiscard):
        """
        Draws route cards and ask the user whether they don't want some of them, and if so, which.
        """
        self.drawRouteCards()

        for i in range(numDiscard):
            print self.tempRoutes
            discard = self.evilInput("To discard a route, y. To keep remaining routes, n. ")
            if discard == "n":
                break
            choice = int(self.evilInput("Choose index 0-2. "))
            self.game.putRouteCards([self.tempRoutes[choice]])
            self.tempRoutes[choice] = None
        for route in self.tempRoutes:
            if route != None:
                self.routes.append(route)
        print self.routes

    def playTrains(self, edge):
        """
        Allow user to determine which edge to place trains, using which cards.
        """
        rainbowChosen = False
        rainbows = 0
        while not rainbowChosen and self.hand["rainbow"] > 0:
            rainbows = int(self.evilInput("how many rainbow cards do you want to use? " ))
            if (self.hand["rainbow"] >= rainbows):
                rainbowChosen = True
        print "rainbows ", rainbows
        if edge.color == None:
            colorChosen = False
            while (not colorChosen):
                pick = self.evilInput("choose color ")
                if (self.hand[pick] >= edge.price - rainbows):
                    edge.owner = self.index
                    self.hand[pick] -= (edge.price - rainbows)
                    self.game.discards[pick] += (edge.price - rainbows)
                    self.hand["rainbow"] -= rainbows
                    self.game.discards["rainbow"] += rainbows
                    colorChosen = True
                    self.score += Game.scoring[edge.price]
                    self.numTrains -= edge.price
                elif pick == "quit":
                    self.takeAction()
                    break
        elif (self.hand[edge.color] >= (edge.price - rainbows)):
            edge.owner = self.index
            self.hand[edge.color] -= (edge.price - rainbows)
            self.game.discards[edge.color] += (edge.price - rainbows)
            self.hand["rainbow"] -= rainbows
            self.game.discards["rainbow"] += rainbows
            self.score += Game.scoring[edge.price]
            self.numTrains -= edge.price
        else:
            print self.hand[edge.color], (edge.price - rainbows)
            self.takeAction()

    def takeAction(self):
        """
        Allow player to choose an action to take
        """
        # Display current board
        visual = open('visual', 'w')
        visual.write(str(self.game.graph))
        visual.close()
        subprocess.call(["fdp", "-Tpdf", "-oboard.pdf", "visual"])
        #subprocess.call(["xdg-open", "board.pdf"])

        print ""
        print ""
        print "riytse ", self.routes
        print "player ", self.index, Game.colors[self.index], "score ", self.score
        print "hand", self.hand
        print "score", map(lambda p: p.score, self.game.players)
        print "hands", map(lambda p: sum(p.hand.values()), self.game.players)
        print "face up", game.revealedTrains
        print "trains", map(lambda p: p.numTrains, self.game.players)
        action = self.evilInput("1 to draw trains, 2 to draw routes, 3 to place trains ")
        if action == "1":
            # Draw 2 cards either face up or from the deck. If one draws a rainbow
            # face up, one may only draw one card.
            for i in range(2):
                print "face up", game.revealedTrains
                pick = self.evilInput("1 to draw randomly, 2 to draw faceup ")
                if pick == "1":
                    self.drawTrainCard()
                if pick == "2":
                    index = int(self.evilInput("Index 0-4 of desired train "))
                    color = self.drawFaceUp(index, i)
                    if color == "rainbow":
                        break
                print "hand", self.hand
            print "face up", game.revealedTrains
        elif action == "2":
            # Get 3 new routes. Discard up to 2.
            self.pickRouteCards(2)
            print self.routes
        elif action == "3":
            # Place trains on the board
            index = int(self.evilInput("Provide the index of the edge you want to put trains on "))
            self.playTrains(self.game.graph.edges[index])
            visual = open('visual', 'w')
            visual.write(str(self.game.graph))
            visual.close()
            print "hand ", self.hand
            subprocess.call(["fdp", "-Tpdf", "-oboard.pdf", "visual"])
        else:
            print "Not an option."
            self.takeAction()

class DummyPlayer(ShoddyUIPlayer):
    """
    Fake player which as much as possible avoids impacting the game by doing
    literally nothing and just causes the game to pause during each turn.
    """
    def __init__(self, game, index):
        self.game = game
        self.index = index
        self.hand = {j: 0 for j in game.discards}
        self.routes = []
        self.score = 0
        self.numTrains = 45

    def pickRouteCards(self, numDiscard):
        pass
    def takeAction(self):
        subprocess.call(["fdp", "-Tpdf", "-oboard.pdf", "visual"])
        self.evilInput("")

class Graph:
    """
    Stores the map and connections between cities.
    """
    def __init__(self):
        """
        This should really be read in from a data file.
        """
        self.order = []
        self.nodes = {} #dict mapping names to lists of edges
        self.edges = []
        with open("nodes.txt") as nodes:
            for line in nodes.readlines():
                city = line.strip()
                self.order.append(city)
                self.nodes[city] = []
        #Create each edge, add to both nodes and edges
        with open("edges.txt") as edges:
            for line in edges.readlines():
                city1, city2, color, price = line.split(",")
                if color == "grey":
                    color = None
                price = int(price)
                edge = Edge(color, int(price), (city1, city2))
                self.edges.append(edge)
                self.nodes[city1].append(edge)
                self.nodes[city2].append(edge)

    def __str__(self):
        """
        Generates input for graphviz
        """
        s = []
        for node in self.order:
            s.append('<%s>;' % node);
        for j, edge in enumerate(self.edges):
            style = {}
            style["label"] = "%s=%s/%s" % (j, edge.price, edge.owner)
            style["len"] = 2*edge.price

            if edge.owner == None:
                style["style"] = "dashed"
                style["color"] = edge.color or "grey"
            else:
                style["color"] = Game.colors[edge.owner] or "grey"
            style["fontcolor"] = style["color"]
            styling = ",".join("%s=<%s>" % (key, style[key]) for key in style)
            s.append('<%s> -- <%s> [%%s];' % edge.nodes % styling)
        return "graph {\n" + "\n".join(s) + "\n}"

class Edge:
    """
    Fancy struct for edge information. This can't just be a tuple because the owner is mutable
    """
    def __init__(self, color, price, nodes, owner = None):
        self.color = color
        self.price = price
        self.nodes = nodes
        self.owner = owner

    def __str__(self):
        return str((self.color, self.price, self.nodes, self.owner))
    def __repr__(self):
        return "Edge" + str(self)

if __name__ == "__main__":
    import sys
    # set seed for deterministic testing
    if sys.argv[1] != "-":
        random.seed(int(sys.argv[1]))
    classes = [globals()[k] for k in sys.argv[2:]]
    game = Game(classes)
    game.playGame()
