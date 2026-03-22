local function log(msg, el)
  local txt = ""
  if el and el.content then
    txt = pandoc.utils.stringify(el.content)
  end
  io.stderr:write("[LOG] " .. msg .. ": " .. txt .. "\n")
end

-- Заголовки
function Header(el)
  if el.identifier ~= "" or #el.classes > 0 then
    log("Header with attrs", el)
  end
  el.identifier = ""
  el.classes = {}
  el.attributes = {}
  return el
end

-- Div
function Div(el)
  if el.identifier ~= "" or #el.classes > 0 then
    log("Div with attrs", el)
  end
  for _, cls in ipairs(el.classes) do
    if cls:match("^sgc%-toc") or cls == "cover" then
      log("Drop Div", el)
      return {}
    end
  end
  el.identifier = ""
  el.classes = {}
  el.attributes = {}
  return pandoc.Div(el.content)
end

-- Span
function Span(el)
  if #el.content == 0 then
    log("Drop empty Span", el)
    return {}
  end
  if el.identifier ~= "" or #el.classes > 0 then
    log("Span with attrs", el)
  end
  el.identifier = ""
  el.classes = {}
  el.attributes = {}
  return el
end

-- Link
function Link(el)
  if #el.content == 0 then
    log("Drop empty Link", el)
    return {}
  end
  if el.identifier ~= "" or #el.classes > 0 then
    log("Link with attrs", el)
  end
  el.identifier = ""
  el.classes = {}
  el.attributes = {}
  return el
end

-- Image
function Image(el)
  if el.identifier ~= "" or #el.classes > 0 then
    log("Image with attrs", el)
  end
  el.identifier = ""
  el.classes = {}
  el.attributes = {}
  el.title = ""
  return el
end

-- RawInline/RawBlock
function RawInline(el)
  log("Drop RawInline", el)
  return {}
end

function RawBlock(el)
  log("Drop RawBlock", el)
  return {}
end