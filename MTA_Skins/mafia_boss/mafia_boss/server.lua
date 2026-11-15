-- Demonstration command available to all players. Remove from meta.xml if unwanted.
addCommandHandler("mafia", function(player)
    if not isElement(player) or getElementType(player) ~= "player" then return end
    if setElementModel(player, 124) then
        outputChatBox("mafia_boss skin selected (ID 124).", player, 170, 200, 230)
    end
end)
