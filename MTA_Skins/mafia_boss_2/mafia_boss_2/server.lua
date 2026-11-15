-- Demonstration command available to all players. Remove from meta.xml if unwanted.
addCommandHandler("mafia2", function(player)
    if not isElement(player) or getElementType(player) ~= "player" then return end
    if setElementModel(player, 186) then
        outputChatBox("mafia_boss_2 skin selected (ID 186).", player, 170, 200, 230)
    end
end)
