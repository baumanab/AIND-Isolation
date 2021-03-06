"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def weighted_openmovediff_score(game, player, weight=2):
    """This evaluation function outputs a score equal to the difference in the number
    of moves available to the two players. Contains an optional weight parameter
    as a multiplier to the  opponents score.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    weight: int
        Optional int argument which weights the oppoents moves by opp_moves * weight.
        This functions to penalize choices where the opponent has more moves.

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves * weight)


def center_weighted_moves(game, player, weight= 2, center_weight= 2):
    """This evaluation function outputs a score based on weighted difference of
    the difference in own moves and opponent moves, further weighted to favor center_weight
    row and column squares.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    weight: int
        Optional int argument which weights the oppoents moves by opp_moves * weight.
        This functions to penalize choices where the opponent has more moves.

    center_weight: int
        Optional int argument which further weights center moves.


    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_col= math.ceil(game.width/2.)
    center_row= math.ceil(game.height/2.)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    num_own_moves= len(own_moves)
    num_opp_moves= len(opp_moves)

    opp_weight, own_weight= weight,1

    for move in own_moves:
        if move[0]== center_row or move[1]== center_col:
            own_weight *= center_weight

    for move in opp_moves:
        if move[0]== center_row or move[1]== center_col:
            opp_weight *= center_weight

    return float((num_own_moves * own_weight) - (num_opp_moves * opp_weight))


def centerdecay_weighted_moves(game, player, weight= 2, center_weight= 2):
        """This evaluation function outputs a score based on weighted difference of
        the difference in own moves and opponent moves, further weighted to favor center_weight
        row and column squares. An additional decay factor has been added which
        decreases center weighting as a function of the number of unblocked squares.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

        weight: int
            Optional int argument which weights the oppoents moves by opp_moves * weight.
            This functions to penalize choices where the opponent has more moves.

        center_weight: int
            Optional int argument which further weights center moves.


        Returns
        ----------
        float
            The heuristic value of the current game state
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("inf")

        center_col= math.ceil(game.width/2.)
        center_row= math.ceil(game.height/2.)

        own_moves = game.get_legal_moves(player)
        opp_moves = game.get_legal_moves(game.get_opponent(player))
        num_own_moves= len(own_moves)
        num_opp_moves= len(opp_moves)

        initial_moves_available= float(game.width * game.height)

        num_blank_spaces= len(game.get_blank_spaces())

        decay_factor= num_blank_spaces/initial_moves_available

        opp_weight, own_weight= weight,1

        for move in own_moves:
            if move[0]== center_row or move[1]== center_col:
                own_weight *= (center_weight * decay_factor)

        for move in opp_moves:
            if move[0]== center_row or move[1]== center_col:
                opp_weight *= (center_weight * decay_factor)

        return float((num_own_moves * own_weight) - (num_opp_moves * opp_weight))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return centerdecay_weighted_moves(game, player)



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # initialize move
        move= None # defensive

        if not legal_moves: return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # initiate search depth trying out single line flow control on simple
            # statments to save space/scrolling
            depth= 1 if self.iterative else self.search_depth

            # conduct iterative DFS or Full DFS depending on flag
            if self.iterative:
                while True: # loop "forever" if timeout exception called and moved returned
                    if self.method== 'minimax':
                        score, move= self.minimax(game, depth)
                    elif self.method== 'alphabeta':
                        score, move= self.alphabeta(game, depth)
                    depth += 1

            else:
                if self.method== 'minimax':
                    score, move= self.minimax(game, depth)
                elif self.method== 'alphabeta':
                    score, move= self.alphabeta(game, depth)
                return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # implementation adapted from
        # http://aima.cs.berkeley.edu/python/games.html
        # http://giocc.com/concise-implementation-of-minimax-through-higher-order-functions.html
        """
        Notes on what we are doing for future reference. Recursive tree traversal
        where root is set by initial flag of max_player= True, calculating score
        starting at depth= -1 (we don't score node== depth because we don't Generate
        succesors from that node). We take the max of min nodes and min of max nodes
        switching minimizing and maximizing player by passing True or False into
        the self.minimax recursive calls.
        """
        # handle depth == 0
        if depth== 0: return self.score(game, self), None

        # define legal moves for active player
        legal_moves= game.get_legal_moves(game.active_player)

        # handle terminal game state
        if not legal_moves: return self.score(game, self), (-1, -1)

        # initialize best_move
        best_move= None
        # recursive call to minimax with player switch
        if maximizing_player: # root
            best_score= float('-inf') # initiate at lowest possible value in the universe
            for move in legal_moves:
                next_game_state= game.forecast_move(move)
                # get min scores
                score, _= self.minimax(next_game_state, depth - 1, False)
                # get max of min
                if score > best_score:
                    best_score= score
                    best_move= move

        else: # opponent @ not root
            best_score= float('inf') # initiate at highest possible value in the universe
            for move in legal_moves:
                next_game_state= game.forecast_move(move)
                # get max scores
                score, _= self.minimax(next_game_state, depth - 1, True)
                # get min of max
                if score < best_score:
                    best_score= score
                    best_move= move

        return best_score, best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # note for minimax we iterate over taking the min of the max and the
        # max of the min.  Alpha is the max max and beta is the min min (they are the bounds)
        # for each node. We compare the maximizing players current max min to the node max min
        # and the minimizing players node min max to the current min max, and prune
        # when the alpha is >= to the return of the min player and when beta is
        # <= to the return of the max player.

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # handle depth == 0
        if depth== 0: return self.score(game, self), None

        # define legal moves for active player
        legal_moves= game.get_legal_moves(game.active_player)

        # handle terminal game state
        if not legal_moves: return self.score(game, self), (-1, -1)

        # initialize best_move
        best_move= None
        # recursive call to minimax with player switch
        if maximizing_player: # root
            best_score= float('-inf') # initiate at lowest possible value in the universe
            for move in legal_moves:
                next_game_state= game.forecast_move(move)
                # get min scores
                score, _= self.alphabeta(next_game_state, depth - 1, alpha, beta, False)
                # get max of min
                if score > best_score:
                    best_score= score
                    best_move= move
                # update alpha from current (initial is inf)
                alpha= max(alpha, score)
                if score >= beta: break


        else: # opponent @ not root
            best_score= float('inf') # initiate at highest possible value in the universe
            for move in legal_moves:
                next_game_state= game.forecast_move(move)
                # get max scores
                score, _= self.alphabeta(next_game_state, depth - 1,alpha, beta, True)
                # get min of max
                if score < best_score:
                    best_score= score
                    best_move= move
                # update beta from curent (initial is -inf)
                beta= min(beta, score)
                if score <= alpha: break

        return best_score, best_move
