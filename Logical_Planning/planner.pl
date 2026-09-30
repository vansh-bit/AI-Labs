% Prolog Knowledge Base for Warehouse Logical Planning
% Task 6: Connected Locations (Facts)
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Task 6: Movement Rule
can_move(X, Y) :- connected(X, Y).

% Task 7: Plan Step Verifier Rule
valid_move(X, Y) :- connected(X, Y).

% Task 8: Logical Implication Chain
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
