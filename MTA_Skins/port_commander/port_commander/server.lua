-- Demonstration command available to all players. Remove from meta.xml if unwanted.
addCommandHandler("portchief", function(player)
    if not isElement(player) or getElementType(player) ~= "player" then return end
    if setElementModel(player, 281) then
        outputChatBox("port_commander skin selected (ID 281).", player, 170, 200, 230)
    end
end)
